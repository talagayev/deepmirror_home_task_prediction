from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
from dimorphite_dl import protonate_smiles

from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass(frozen=True)
class StandardizeResult:
    smiles_std: Optional[str]
    reason: Optional[str] = None


def standardize_smiles(
    smiles: object,
    *,
    keep_organic: bool = True,
    uncharge: bool = True,
    canonical_tautomer: bool = True,
    keep_isomeric: bool = True,
    protonate: bool = False,
    ph: float = 7.4,
) -> "StandardizeResult":

    # Treat None/NaN/empty as empty_smiles
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        return StandardizeResult(None, "empty_smiles")
    s = str(smiles).strip()
    if not s:
        return StandardizeResult(None, "empty_smiles")

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return StandardizeResult(None, "invalid_smiles")

    try:
        # Here AI first suggested Cleanup --> protonation --> Salt removal --> tautomer.
        # Questionable order, considering then it would do the protonation on salts
        # First step is the standardization of molecule to remove strange cases
        mol = rdMolStandardize.Cleanup(mol)

        # remove salts. don't remove everything
        # Here it is also an option to use rdMolStandardize.LargestFragmentChooser() if there are multiple Smiles and also for the SaltRemover to select the SMARTS patterns
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)
        if (
            mol is None or mol.GetNumAtoms() == 0
        ):  # Cover case if everything was still removed
            return StandardizeResult(None, "empty_after_desalt")

        # Here we now come to the case that we may have a mix of protonated/salt molecules and non protonated molecules.
        # We need to treat them in a similar style, so we either protonate or remove charges, also still an option to not
        # do any of those and keep them as it is
        # Here is the tool publication https://doi.org/10.1186/s13321-019-0336-9
        # Here I was thinking if it should be tautomer --> protonation or protonation --> tautomer
        if protonate:
            uncharge = False  # protonation and uncharging are contradictory in practice
            base_smiles = Chem.MolToSmiles(
                mol, canonical=True, isomericSmiles=keep_isomeric
            )

            # Here there are multiple settings, like the precision, which increases the amount of correct, but also incorrect predictions
            # Here I would keep defaults, due to the paper suggestion
            # Overall the tool gives you multiple variants
            variants = protonate_smiles(
                base_smiles,
                ph_min=ph,
                ph_max=ph,
                max_variants=1,  # Here I selected 1 to not get multiple variants for the same molecule. problem is that if it has multiple variants it may not give here the same variant
            )
            if not variants:
                return StandardizeResult(None, "protonate_no_variants")

            mol = Chem.MolFromSmiles(variants[0])
            if mol is None:
                return StandardizeResult(None, "protonate_invalid_variant")

        elif uncharge:
            mol = rdMolStandardize.Uncharger().uncharge(mol)

        # Here let's make the canonical tautomer selection optional.
        # Overall another option is probably to use something like Weasel, but this is 3D so you would need to use conformer generators
        if canonical_tautomer:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)

        smiles_std = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=keep_isomeric)
        return StandardizeResult(smiles_std, None)

    except Exception as e:
        return StandardizeResult(None, f"standardize_error:{type(e).__name__}")
