from typing import Dict, Type
from pipelines.IPipeline import IPipeline
from pipelines.github.pipeline import GitHubPipeline

class PipelineRegistry:
    """
    Registry for managing different pipeline implementations.
    """
    
    def __init__(self):
        self._pipelines: Dict[str, Type[IPipeline]] = {}
        self._register_default_pipelines()
    
    def _register_default_pipelines(self):
        """Register default pipeline implementations."""
        self.register("github", GitHubPipeline)
    
    def register(self, name: str, pipeline_class: Type[IPipeline]):
        """
        Register a pipeline implementation.
        
        Args:
            name: The name to register the pipeline under
            pipeline_class: The pipeline class to register
        """
        self._pipelines[name] = pipeline_class
    
    def get_pipeline(self, name: str) -> Type[IPipeline]:
        """
        Get a registered pipeline class by name.
        
        Args:
            name: The name of the pipeline to retrieve
            
        Returns:
            Type[IPipeline]: The pipeline class
            
        Raises:
            KeyError: If the pipeline name is not registered
        """
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not found. Available pipelines: {list(self._pipelines.keys())}")
        
        return self._pipelines[name]
    
    def list_pipelines(self) -> list[str]:
        """
        List all registered pipeline names.
        
        Returns:
            list[str]: List of registered pipeline names
        """
        return list(self._pipelines.keys())

# Global registry instance
pipeline_registry = PipelineRegistry()
