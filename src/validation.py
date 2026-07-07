"""
SMILES Validation and Model Name Parsing
=========================================

Centralized utilities for validating chemical SMILES strings
and parsing model/quantization names from file naming conventions.
"""

from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog("rdApp.*")


def validate_smiles(smiles_list: list[str]) -> list[str]:
    """
    Validate a list of SMILES strings using RDKit.

    For each SMILES, attempts to parse and canonicalize it.
    Invalid SMILES are silently dropped.

    Args:
        smiles_list: List of raw SMILES strings.

    Returns:
        List of canonical SMILES strings (only valid ones).
    """
    valid = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid.append(Chem.MolToSmiles(mol))
        except Exception:
            pass
    return valid


def parse_model_name(full_name: str) -> tuple[str, str]:
    """
    Extract the base model name and quantization from a full model identifier.

    Examples:
        "gemma-4-31B-Q8"  -> ("gemma-4-31B", "Q8")
        "Qwen3.6-35B-BF16" -> ("Qwen3.6-35B", "BF16")
        "gpt-oss-120B-Q4"  -> ("gpt-oss-120B", "Q4")

    Args:
        full_name: The full model identifier string.

    Returns:
        Tuple of (model_base, quantization).
    """
    parts = full_name.rsplit("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return full_name, "unknown"


def parse_replica_name(replica_str: str) -> tuple[str, str, int]:
    """
    Parse a replica identifier string into model, quantization, and replica number.

    Example:
        "gemma-4-31B-Q8_replica_8" -> ("gemma-4-31B", "Q8", 8)

    Args:
        replica_str: The replica identifier (e.g., from CSV filename or column).

    Returns:
        Tuple of (model_base, quantization, replica_number).
    """
    # Split on "_replica_" to separate model+quant from replica number
    model_quant, replica_num_str = replica_str.split("_replica_")
    model_base, quantization = parse_model_name(model_quant)
    return model_base, quantization, int(replica_num_str)
