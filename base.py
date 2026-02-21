# bank/core/base.py
"""Base classes for all modules"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from .validation import ChemEngError

class SolverBase(ABC):
    """Base class for all solvers"""
    
    @abstractmethod
    def solve(self) -> Dict[str, Any]:
        """Main solving method"""
        pass
    
    def validate(self) -> None:
        """Validate inputs before solving"""
        pass
    
    def summary(self) -> Dict[str, Any]:
        """Return summary of results"""
        return self.solve()

@dataclass
class SpecificationBase:
    """Base class for all specifications"""
    
    def validate(self) -> None:
        """Validate specification parameters"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}