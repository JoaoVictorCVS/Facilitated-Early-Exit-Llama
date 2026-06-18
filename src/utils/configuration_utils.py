from typing import Any, Dict
import yaml
import logging

from .logging_utils import *

logger = logging.getLogger(__name__)

class GenericConfig:
    """Configuration class with YAML serialization support."""
    
    def __init__(self, default_config):
        """Initialize config with default values.
        
        Parameters
        ----------
        default_config : dict
            Dictionary with default configuration values.
        """
        self.config: Dict[str, Any] = default_config
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load config from YAML file, merging with defaults.
        
        Parameters
        ----------
        filepath : str
            Path to YAML configuration file.
            
        Returns
        -------
        Config
            Config object with merged values.
        """
        config = cls()
        with open(filepath, 'r') as f:
            loaded = yaml.safe_load(f)
        config._merge(config.config, loaded)
        return config
    
    def _merge(self, base: Dict, update: Dict):
        """Recursively merge update dict into base dict.
        
        Parameters
        ----------
        base : dict
            Base configuration dictionary.
        update : dict
            Dictionary with updates to merge.
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge(base[key], value)
            else:
                if not key in base:
                    logger.warning(f"Unknown config key '{key}' found in the loaded configuration.")
                base[key] = value
    
    def dump(self, filepath: str):
        """Save config to YAML file.
        
        Parameters
        ----------
        filepath : str
            Path where YAML file will be written.
        """
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def to_string(self):
        """Return config as a YAML-formatted string.
        
        Returns
        -------
        str
            YAML-formatted configuration string.
        """
        return yaml.dump(self.config, default_flow_style=False)

    def __getitem__(self, key: str) -> Any:
        """Retrieve config value by key.
        
        Parameters
        ----------
        key : str
            Configuration key.
            
        Returns
        -------
        Any
            Configuration value.
        """
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set config value by key.
        
        Parameters
        ----------
        key : str
            Configuration key.
        value : Any
            Value to set.
        """
        self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default fallback.
        
        Parameters
        ----------
        key : str
            Configuration key.
        default : Any, optional
            Default value if key is not found (default is None).
            
        Returns
        -------
        Any
            Configuration value or default if key is not found.
        """
        return self.config.get(key, default)