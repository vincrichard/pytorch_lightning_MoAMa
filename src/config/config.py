import ast
import copy
import difflib
import os
import os.path as osp
import re
from collections import OrderedDict, abc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

from addict import Dict
from rich.console import Console
from rich.text import Text
from yapf.yapflib.yapf_api import FormatCode

from .lazy import LazyAttr, LazyObject
from .utils import (
    ConfigParsingError,
    InstanciationError,
    get_installed_path,
    ImportTransformer,
    _gather_abs_import_lazyobj,
)


BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
DEPRECATION_KEY = "_deprecation_"
RESERVED_KEYS = ["filename", "text", "pretty_text", "env_variables"]


def _is_builtin_module(*args, **kwargs):
    return False


class ConfigDict(Dict):
    """A dictionary for config which has the same interface as python's built-
    in dictionary and can be used as a normal dictionary.

    The Config class would transform the nested fields (dictionary-like fields)
    in config file into ``ConfigDict``.

    If the class attribute ``lazy``  is ``False``, users will get the
    object built by ``LazyObject`` or ``LazyAttr``, otherwise users will get
    the ``LazyObject`` or ``LazyAttr`` itself.

    The ``lazy`` should be set to ``True`` to avoid building the imported
    object during configuration parsing, and it should be set to False outside
    the Config to ensure that users do not experience the ``LazyObject``.
    """

    lazy = False

    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, "__parent", kwargs.pop("__parent", None))
        object.__setattr__(__self, "__key", kwargs.pop("__key", None))
        object.__setattr__(__self, "__frozen", False)
        for arg in args:
            if not arg:
                continue
            # Since ConfigDict.items will convert LazyObject to real object
            # automatically, we need to call super().items() to make sure
            # the LazyObject will not be converted.
            if isinstance(arg, ConfigDict):
                for key, val in dict.items(arg):
                    __self[key] = __self._hook(val)
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in dict.items(kwargs):
            __self[key] = __self._hook(val)

    @property
    def filename(self) -> str:
        """get file name of config."""
        return self._filename

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
            if isinstance(value, (LazyAttr, LazyObject)) and not self.lazy:
                value = value.build()
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no " f"attribute '{name}'")
        except Exception as e:
            raise e
        else:
            return value

    @classmethod
    def _hook(cls, item):
        # avoid to convert user defined dict to ConfigDict.
        if type(item) in (dict, OrderedDict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __setattr__(self, name, value):
        value = self._hook(value)
        return super().__setattr__(name, value)

    def __setitem__(self, name, value):
        value = self._hook(value)
        return super().__setitem__(name, value)

    def __getitem__(self, key):
        return self.build_lazy(super().__getitem__(key))

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in super().items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def __copy__(self):
        other = self.__class__()
        for key, value in super().items():
            other[key] = value
        return other

    copy = __copy__

    def __iter__(self):
        # Implement `__iter__` to overwrite the unpacking operator `**cfg_dict`
        # to get the built lazy object
        return iter(self.keys())

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get the value of the key. If class attribute ``lazy`` is True, the
        LazyObject will be built and returned.

        Args:
            key (str): The key.
            default (any, optional): The default value. Defaults to None.

        Returns:
            Any: The value of the key.
        """
        return self.build_lazy(super().get(key, default))

    def pop(self, key, default=None):
        """Pop the value of the key. If class attribute ``lazy`` is True, the
        LazyObject will be built and returned.

        Args:
            key (str): The key.
            default (any, optional): The default value. Defaults to None.

        Returns:
            Any: The value of the key.
        """
        return self.build_lazy(super().pop(key, default))

    def update(self, *args, **kwargs) -> None:
        """Override this method to make sure the LazyObject will not be built
        during updating."""
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError("update only accept one positional argument")
            # Avoid to used self.items to build LazyObject
            for key, value in dict.items(args[0]):
                other[key] = value

        for key, value in dict(kwargs).items():
            other[key] = value
        for k, v in other.items():
            if (k not in self) or (not isinstance(self[k], dict)) or (not isinstance(v, dict)):
                self[k] = self._hook(v)
            else:
                self[k].update(v)

    def build_lazy(self, value: Any) -> Any:
        """If class attribute ``lazy`` is False, the LazyObject will be built
        and returned.

        Args:
            value (Any): The value to be built.

        Returns:
            Any: The built value.
        """
        if isinstance(value, (LazyAttr, LazyObject)) and not self.lazy:
            value = value.build()
        return value

    def build(self, **kwargs) -> Any:
        if "type" not in self and "type" not in kwargs:
            raise KeyError('`cfg` or `default_args` must contain the key "type", ' f"but got {self}\n{kwargs}")

        parameters = self.copy()
        parameters.update(kwargs)
        args = list(parameters.pop("args", []))

        for key, value in parameters.items():
            if isinstance(value, ConfigDict) and hasattr(value, "type"):
                parameters[key] = value.build()

        for i, value in enumerate(args):
            if isinstance(value, ConfigDict) and hasattr(value, "type"):
                args[i] = value.build()

        class_instanciatior = parameters.pop("type")

        try:
            return class_instanciatior(*args, **parameters)
        except Exception as e:
            raise InstanciationError(
                f"The class {class_instanciatior.__name__} failed to be created with the arguments {args}"
            ) from e

    def values(self):
        """Yield the values of the dictionary.

        If class attribute ``lazy`` is False, the value of ``LazyObject`` or
        ``LazyAttr`` will be built and returned.
        """
        values = []
        for value in super().values():
            values.append(self.build_lazy(value))
        return values

    def items(self):
        """Yield the keys and values of the dictionary.

        If class attribute ``lazy`` is False, the value of ``LazyObject`` or
        ``LazyAttr`` will be built and returned.
        """
        items = []
        for key, value in super().items():
            items.append((key, self.build_lazy(value)))
        return items

    def merge(self, other: dict):
        """Merge another dictionary into current dictionary.

        Args:
            other (dict): Another dictionary.
        """
        default = object()

        def _merge_a_into_b(a, b):
            if isinstance(a, dict):
                if not isinstance(b, dict):
                    a.pop(DELETE_KEY, None)
                    return a
                if a.pop(DELETE_KEY, False):
                    b.clear()
                all_keys = list(b.keys()) + list(a.keys())
                return {
                    key: _merge_a_into_b(a.get(key, default), b.get(key, default))
                    for key in all_keys
                    if key != DELETE_KEY
                }
            else:
                return a if a is not default else b

        merged = _merge_a_into_b(copy.deepcopy(other), copy.deepcopy(self))
        self.clear()
        for key, value in merged.items():
            self[key] = value

    def __eq__(self, other):
        if isinstance(other, ConfigDict):
            return other.to_dict() == self.to_dict()
        elif isinstance(other, dict):
            return {k: v for k, v in self.items()} == other
        else:
            return False

    def _to_lazy_dict(self):
        """Convert the ConfigDict to a normal dictionary recursively, and keep
        the ``LazyObject`` or ``LazyAttr`` object not built."""

        def _to_dict(data):
            if isinstance(data, ConfigDict):
                return {key: _to_dict(value) for key, value in Dict.items(data)}
            elif isinstance(data, dict):
                return {key: _to_dict(value) for key, value in data.items()}
            elif isinstance(data, (list, tuple)):
                return type(data)(_to_dict(item) for item in data)
            else:
                return data

        return _to_dict(self)

    def to_dict(self):
        """Convert the ConfigDict to a normal dictionary recursively, and
        convert the ``LazyObject`` or ``LazyAttr`` to string."""
        return self._lazy2string(self, dict_type=dict)

    def _lazy2string(self, cfg_dict, dict_type=None):
        if isinstance(cfg_dict, dict):
            dict_type = dict_type or type(cfg_dict)
            return dict_type({k: self._lazy2string(v, dict_type) for k, v in dict.items(cfg_dict)})
        elif isinstance(cfg_dict, (tuple, list)):
            return type(cfg_dict)(self._lazy2string(v, dict_type) for v in cfg_dict)
        elif isinstance(cfg_dict, (LazyAttr, LazyObject)):
            return f"{cfg_dict.module}.{str(cfg_dict)}"
        else:
            return cfg_dict


class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml.
    ``Config.fromfile`` can parse a dictionary from a config file, then
    build a ``Config`` instance with the dictionary.
    The interface is the same as a dict object and also allows access config
    values as attributes.

    Args:
        cfg_dict (dict, optional): A config dictionary. Defaults to None.
        cfg_text (str, optional): Text of config. Defaults to None.
        filename (str or Path, optional): Name of config file.
            Defaults to None.
        format_python_code (bool): Whether to format Python code by yapf.
            Defaults to True.

    Here is a simple example:

    Examples:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/username/projects/mmengine/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/username/projects/mmengine/tests/data/config/a.py]
        :"
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"

    You can find more advance usage in the `config tutorial`_.

    .. _config tutorial: https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html
    """  # noqa: E501

    def __init__(
        self,
        cfg_dict: dict = None,
        cfg_text: Optional[str] = None,
        filename: Optional[Union[str, Path]] = None,
        env_variables: Optional[dict] = None,
        format_python_code: bool = True,
    ):
        filename = str(filename) if isinstance(filename, Path) else filename
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError("cfg_dict must be a dict, but " f"got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        if not isinstance(cfg_dict, ConfigDict):
            cfg_dict = ConfigDict(cfg_dict)
        super().__setattr__("_cfg_dict", cfg_dict)
        super().__setattr__("_filename", filename)
        super().__setattr__("_format_python_code", format_python_code)
        if not hasattr(self, "_imported_names"):
            super().__setattr__("_imported_names", set())

        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, encoding="utf-8") as f:
                text = f.read()
        else:
            text = ""
        super().__setattr__("_text", text)
        if env_variables is None:
            env_variables = dict()
        super().__setattr__("_env_variables", env_variables)

    @staticmethod
    def fromfile(
        filename: Union[str, Path],
        # use_predefined_variables: bool = True,
        # import_custom_modules: bool = True,
        # use_environment_variables: bool = True,
        # lazy_import: Optional[bool] = None,
        format_python_code: bool = True,
    ) -> "Config":
        """Build a Config instance from config file.

        Args:
            filename (str or Path): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.
            import_custom_modules (bool, optional): Whether to support
                importing custom modules in config. Defaults to None.
            use_environment_variables (bool, optional): Whether to use
                environment variables. Defaults to True.
            lazy_import (bool): Whether to load config in `lazy_import` mode.
                If it is `None`, it will be deduced by the content of the
                config file. Defaults to None.
            format_python_code (bool): Whether to format Python code by yapf.
                Defaults to True.

        Returns:
            Config: Config instance built from config file.
        """
        filename = str(filename) if isinstance(filename, Path) else filename
        # if lazy_import is False or lazy_import is None and not Config._is_lazy_import(filename):
        #     cfg_dict, cfg_text, env_variables = Config._file2dict(
        #         filename, use_predefined_variables, use_environment_variables, lazy_import
        #     )
        #     if import_custom_modules and cfg_dict.get("custom_imports", None):
        #         try:
        #             import_modules_from_strings(**cfg_dict["custom_imports"])
        #         except ImportError as e:
        #             err_msg = (
        #                 "Failed to import custom modules from "
        #                 f"{cfg_dict['custom_imports']}, the current sys.path "
        #                 "is: "
        #             )
        #             for p in sys.path:
        #                 err_msg += f"\n    {p}"
        #             err_msg += (
        #                 "\nYou should set `PYTHONPATH` to make `sys.path` "
        #                 "include the directory which contains your custom "
        #                 "module"
        #             )
        #             raise ImportError(err_msg) from e
        #     return Config(
        #         cfg_dict,
        #         cfg_text=cfg_text,
        #         filename=filename,
        #         env_variables=env_variables,
        #     )
        # else:
        # Enable lazy import when parsing the config.
        # Using try-except to make sure ``ConfigDict.lazy`` will be reset
        # to False. See more details about lazy in the docstring of
        # ConfigDict
        ConfigDict.lazy = True
        try:
            cfg_dict, imported_names = Config._parse_lazy_import(filename)
        except Exception as e:
            raise e
        finally:
            # disable lazy import to get the real type. See more details
            # about lazy in the docstring of ConfigDict
            ConfigDict.lazy = False

        cfg = Config(cfg_dict, filename=filename, format_python_code=format_python_code)
        object.__setattr__(cfg, "_imported_names", imported_names)
        return cfg

    def build(self):
        ConfigDict.lazy = False
        for name, value in self.items():
            if isinstance(value, ConfigDict) and hasattr(value, "type"):
                self[name] = value.build()

    @staticmethod
    def _is_lazy_import(filename: str) -> bool:
        if not filename.endswith(".py"):
            return False
        with open(filename, encoding="utf-8") as f:
            codes_str = f.read()
            parsed_codes = ast.parse(codes_str)
        for node in ast.walk(parsed_codes):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == BASE_KEY
            ):
                return False

            if isinstance(node, ast.With):
                expr = node.items[0].context_expr
                if not isinstance(expr, ast.Call) or not expr.func.id == "read_base":  # type: ignore
                    raise ConfigParsingError("Only `read_base` context manager can be used in the " "config")
                return True
            if isinstance(node, ast.ImportFrom):
                # relative import -> lazy_import
                if node.level != 0:
                    return True
                # Skip checking when using `mmengine.config` in cfg file
                if node.module == "mmengine" and len(node.names) == 1 and node.names[0].name == "Config":
                    continue
                if not isinstance(node.module, str):
                    continue
                # non-builtin module -> lazy_import
                if not _is_builtin_module(node.module):
                    return True
            if isinstance(node, ast.Import):
                for alias_node in node.names:
                    if not _is_builtin_module(alias_node.name):
                        return True
        return False

    @staticmethod
    def _parse_lazy_import(filename: str) -> Tuple[ConfigDict, set]:
        """Transform file to variables dictionary.

        Args:
            filename (str): Name of config file.

        Returns:
            Tuple[dict, dict]: ``cfg_dict`` and ``imported_names``.

              - cfg_dict (dict): Variables dictionary of parsed config.
              - imported_names (set): Used to mark the names of
                imported object.
        """
        # In lazy import mode, users can use the Python syntax `import` to
        # implement inheritance between configuration files, which is easier
        # for users to understand the hierarchical relationships between
        # different configuration files.

        # Besides, users can also using `import` syntax to import corresponding
        # module which will be filled in the `type` field. It means users
        # can directly navigate to the source of the module in the
        # configuration file by clicking the `type` field.

        # To avoid really importing the third party package like `torch`
        # during import `type` object, we use `_parse_lazy_import` to parse the
        # configuration file, which will not actually trigger the import
        # process, but simply parse the imported `type`s as LazyObject objects.

        # The overall pipeline of _parse_lazy_import is:
        # 1. Parse the base module from the config file.
        #                       ||
        #                       \/
        #       base_module = ['mmdet.configs.default_runtime']
        #                       ||
        #                       \/
        # 2. recursively parse the base module and gather imported objects to
        #    a dict.
        #                       ||
        #                       \/
        #       The base_dict will be:
        #       {
        #           'mmdet.configs.default_runtime': {...}
        #           'mmdet.configs.retinanet_r50_fpn_1x_coco': {...}
        #           ...
        #       }, each item in base_dict is a dict of `LazyObject`
        # 3. parse the current config file filling the imported variable
        #    with the base_dict.
        #
        # 4. During the parsing process, all imported variable will be
        #    recorded in the `imported_names` set. These variables can be
        #    accessed, but will not be dumped by default.

        with open(filename, encoding="utf-8") as f:
            global_dict = {"LazyObject": LazyObject, "__file__": filename}
            base_dict = {}

            parsed_codes = ast.parse(f.read())
            # get the names of base modules, and remove the
            # `with read_base():'` statement
            base_modules = Config._get_base_modules(parsed_codes.body)
            base_imported_names = set()
            for base_module in base_modules:
                # If base_module means a relative import, assuming the level is
                # 2, which means the module is imported like
                # "from ..a.b import c". we must ensure that c is an
                # object `defined` in module b, and module b should not be a
                # package including `__init__` file but a single python file.
                level = len(re.match(r"\.*", base_module).group())
                if level > 0:
                    # Relative import
                    base_dir = osp.dirname(filename)
                    module_path = osp.join(
                        base_dir, *([".."] * (level - 1)), f'{base_module[level:].replace(".", "/")}.py'
                    )
                else:
                    # Absolute import
                    module_list = base_module.split(".")
                    if len(module_list) == 1:
                        raise ConfigParsingError(
                            "The imported configuration file should not be "
                            f"an independent package {module_list[0]}. Here "
                            "is an example: "
                            "`with read_base(): from mmdet.configs.retinanet_r50_fpn_1x_coco import *`"  # noqa: E501
                        )
                    else:
                        package = module_list[0]
                        root_path = get_installed_path(package)
                        module_path = f"{osp.join(root_path, *module_list[1:])}.py"  # noqa: E501
                if not osp.isfile(module_path):
                    raise ConfigParsingError(
                        f"{module_path} not found! It means that incorrect "
                        "module is defined in "
                        f"`with read_base(): = from {base_module} import ...`, please "  # noqa: E501
                        "make sure the base config module is valid "
                        "and is consistent with the prior import "
                        "logic"
                    )
                _base_cfg_dict, _base_imported_names = Config._parse_lazy_import(module_path)  # noqa: E501
                base_imported_names |= _base_imported_names
                # The base_dict will be:
                # {
                #     'mmdet.configs.default_runtime': {...}
                #     'mmdet.configs.retinanet_r50_fpn_1x_coco': {...}
                #     ...
                # }
                base_dict[base_module] = _base_cfg_dict

            # `base_dict` contains all the imported modules from `base_cfg`.
            # In order to collect the specific imported module from `base_cfg`
            # before parse the current file, we using AST Transform to
            # transverse the imported module from base_cfg and merge then into
            # the global dict. After the ast transformation, most of import
            # syntax will be removed (except for the builtin import) and
            # replaced with the `LazyObject`
            transform = ImportTransformer(global_dict=global_dict, base_dict=base_dict, filename=filename)
            modified_code = transform.visit(parsed_codes)
            modified_code, abs_imported = _gather_abs_import_lazyobj(modified_code, filename=filename)
            imported_names = transform.imported_obj | abs_imported
            imported_names |= base_imported_names
            modified_code = ast.fix_missing_locations(modified_code)
            exec(compile(modified_code, filename, mode="exec"), global_dict, global_dict)

            ret: dict = {}
            for key, value in global_dict.items():
                if key.startswith("__") or key in ["LazyObject"]:
                    continue
                ret[key] = value
            # convert dict to ConfigDict
            cfg_dict = Config._dict_to_config_dict_lazy(ret)

            return cfg_dict, imported_names

    @staticmethod
    def _dict_to_config_dict_lazy(cfg: Union[dict, Any]):
        """Recursively converts ``dict`` to :obj:`ConfigDict`. The only
        difference between ``_dict_to_config_dict_lazy`` and
        ``_dict_to_config_dict_lazy`` is that the former one does not consider
        the scope, and will not trigger the building of ``LazyObject``.

        Args:
            cfg (dict): Config dict.

        Returns:
            ConfigDict: Converted dict.
        """
        # Only the outer dict with key `type` should have the key `_scope_`.
        if isinstance(cfg, dict):
            cfg_dict = ConfigDict()
            for key, value in cfg.items():
                cfg_dict[key] = Config._dict_to_config_dict_lazy(value)
            return cfg_dict
        if isinstance(cfg, (tuple, list)):
            return type(cfg)(Config._dict_to_config_dict_lazy(_cfg) for _cfg in cfg)
        return cfg

    @staticmethod
    def _get_base_modules(nodes: list) -> list:
        """Get base module name from parsed code.

        Args:
            nodes (list): Parsed code of the config file.

        Returns:
            list: Name of base modules.
        """

        def _get_base_module_from_with(with_nodes: list) -> list:
            """Get base module name from if statement in python file.

            Args:
                with_nodes (list): List of if statement.

            Returns:
                list: Name of base modules.
            """
            base_modules = []
            for node in with_nodes:
                assert isinstance(node, ast.ImportFrom), (
                    "Illegal syntax in config file! Only "
                    "`from ... import ...` could be implemented` in "
                    "with read_base()`"
                )
                assert node.module is not None, (
                    "Illegal syntax in config file! Syntax like "
                    "`from . import xxx` is not allowed in `with read_base()`"
                )
                base_modules.append(node.level * "." + node.module)
            return base_modules

        for idx, node in enumerate(nodes):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == BASE_KEY
            ):
                raise ConfigParsingError(
                    "The configuration file type in the inheritance chain "
                    "must match the current configuration file type, either "
                    '"lazy_import" or non-"lazy_import". You got this error '
                    f'since you use the syntax like `_base_ = "{node.targets[0].id}"` '  # noqa: E501
                    "in your config. You should use `with read_base(): ... to` "  # noqa: E501
                    "mark the inherited config file. See more information "
                    "in https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html"  # noqa: E501
                )

            if not isinstance(node, ast.With):
                continue

            expr = node.items[0].context_expr
            if not isinstance(expr, ast.Call) or not expr.func.id == "read_base" or len(node.items) > 1:  # type: ignore
                raise ConfigParsingError("Only `read_base` context manager can be used in the " "config")

            # The original code:
            # ```
            # with read_base():
            #     from .._base_.default_runtime import *
            # ```
            # The processed code:
            # ```
            # from .._base_.default_runtime import *
            # ```
            # As you can see, the if statement is removed and the
            # from ... import statement will be unindent
            for nested_idx, nested_node in enumerate(node.body):
                nodes.insert(idx + nested_idx + 1, nested_node)
            nodes.pop(idx)
            return _get_base_module_from_with(node.body)
        return []

    def __repr__(self):
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self) -> Tuple[dict, Optional[str], Optional[str], dict, bool, set]:
        state = (
            self._cfg_dict,
            self._filename,
            self._text,
            self._env_variables,
            self._format_python_code,
            self._imported_names,
        )
        return state

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)
        super(Config, other).__setattr__("_cfg_dict", self._cfg_dict.copy())

        return other

    copy = __copy__

    def __setstate__(self, state: Tuple[dict, Optional[str], Optional[str], dict, bool, set]):
        super().__setattr__("_cfg_dict", state[0])
        super().__setattr__("_filename", state[1])
        super().__setattr__("_text", state[2])
        super().__setattr__("_env_variables", state[3])
        super().__setattr__("_format_python_code", state[4])
        super().__setattr__("_imported_names", state[5])

    def dump(self, file: Optional[Union[str, Path]] = None):
        """Dump config to file or return config text.

        Args:
            file (str or Path, optional): If not specified, then the object
            is dumped to a str, otherwise to a file specified by the filename.
            Defaults to None.

        Returns:
            str or None: Config text.
        """
        file = str(file) if isinstance(file, Path) else file
        cfg_dict = self.to_dict()
        if file is None:
            if self.filename is None or self.filename.endswith(".py"):
                return self.pretty_text
        with open(file, "w", encoding="utf-8") as f:
            f.write(self.pretty_text)

    @property
    def pretty_text(self) -> str:
        """get formatted python config text."""

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = repr(v)
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list_tuple(k, v, use_mapping=False):
            if isinstance(v, list):
                left = "["
                right = "]"
            else:
                left = "("
                right = ")"

            v_str = f"{left}\n"
            # check if all items in the list are dict
            for item in v:
                if isinstance(item, dict):
                    v_str += f"dict({_indent(_format_dict(item), indent)}),\n"
                elif isinstance(item, tuple):
                    v_str += f"{_indent(_format_list_tuple(None, item), indent)},\n"  # noqa: 501
                elif isinstance(item, list):
                    v_str += f"{_indent(_format_list_tuple(None, item), indent)},\n"  # noqa: 501
                elif isinstance(item, str):
                    v_str += f"{_indent(repr(item), indent)},\n"
                else:
                    v_str += str(item) + ",\n"
            if k is None:
                return _indent(v_str, indent) + right
            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent) + right
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(sorted(input_dict.items(), key=lambda x: str(x[0]))):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{str(k)}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, (list, tuple)):
                    attr_str = _format_list_tuple(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        if self._format_python_code:
            # copied from setup.cfg
            yapf_style = dict(
                based_on_style="pep8",
                blank_line_before_nested_class_or_def=True,
                split_before_expression_after_opening_paren=True,
            )
            try:
                text, _ = FormatCode(text, style_config=yapf_style)
            except:  # noqa: E722
                raise SyntaxError("Failed to format the config file, please " f"check the syntax of: \n{text}")
        return text


@contextmanager
def read_base():
    """Context manager to mark the base config.

    The pure Python-style configuration file allows you to use the import
    syntax. However, it is important to note that you need to import the base
    configuration file within the context of ``read_base``, and import other
    dependencies outside of it.

    You can see more usage of Python-style configuration in the `tutorial`_

    .. _tutorial: https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta
    """  # noqa: E501
    yield


if __name__ == "__main__":
    import torch

    cfg = Config.fromfile("configs/test_config.py")
    optimizer = cfg.optimizer.params = [torch.Tensor([0])]
    a = 0
