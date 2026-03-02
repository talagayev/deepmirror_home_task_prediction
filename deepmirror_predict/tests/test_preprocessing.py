import pytest
from rdkit import Chem

from deepmirror_predict.data_preprocession.preprocessing import standardize_smiles


def _formal_charge(smiles: str) -> int:
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    return int(Chem.GetFormalCharge(m))


def test_empty_none():
    # Check if we cover the case that smiles is None
    r = standardize_smiles(None)
    assert r.smiles_std is None
    assert r.reason == "empty_smiles"


def test_empty_whitespace():
    # Check if we cover the case where it is empty space
    r = standardize_smiles("   ")
    assert r.smiles_std is None
    assert r.reason == "empty_smiles"


def test_empty_nan():
    # Check Nan case
    r = standardize_smiles(float("nan"))
    assert r.smiles_std is None
    assert r.reason == "empty_smiles"


def test_invalid_smiles():
    # Check case where we have something that is not smiles
    r = standardize_smiles("not_a_smiles")
    assert r.smiles_std is None
    assert r.reason == "invalid_smiles"


def test_desalt_and_uncharge_sodium_acetate():
    # Sodium acetate -> should end up neutral after uncharge
    r = standardize_smiles("CC(=O)[O-].[Na+]", protonate=False, uncharge=True)
    assert r.reason is None
    assert r.smiles_std is not None
    assert r.smiles_std in {"CC(=O)O"}
    assert _formal_charge(r.smiles_std) == 0


def test_protonate_nonionizable_benzene_is_stable():
    # Check that there are no strange protonation artifacts
    r = standardize_smiles("c1ccccc1", protonate=True, ph=7.4)
    assert r.reason is None
    assert r.smiles_std in {"c1ccccc1"}


def test_protonate_disables_uncharge_keeps_charge_when_applicable():
    # Start with a salt; remove salt and protonate, retaining acetate
    r = standardize_smiles("CC(=O)[O-].[Na+]", protonate=True, uncharge=True, ph=7.4)
    assert r.reason is None
    assert r.smiles_std is not None
    assert r.smiles_std in {"CC(=O)[O-]"}
    assert _formal_charge(r.smiles_std) != 0


def test_desalt_and_uncharge_sodium_acetate_exact_smiles():
    # Sodium acetate -> acetic acid after uncharge
    r = standardize_smiles(
        "CC(=O)[O-].[Na+]", protonate=False, uncharge=True, canonical_tautomer=True
    )
    assert r.reason is None
    assert r.smiles_std == "CC(=O)O"
    assert _formal_charge(r.smiles_std) == 0


def test_protonate_guanidine_cation_exact_smiles():
    # Guandine case --> should be protonated
    r = standardize_smiles("NC(N)=N", protonate=True, ph=7.4, canonical_tautomer=True)
    assert r.reason is None
    assert r.smiles_std == "NC(N)=[NH2+]"
    assert _formal_charge(r.smiles_std) == 1


def test_protonate_acetic_acid_anion_exact_smiles():
    # Acetic acid pKa ~4.8, so at pH 7.4 it should be deprotonated
    r = standardize_smiles("CC(=O)O", protonate=True, ph=7.4, canonical_tautomer=True)
    assert r.reason is None
    assert r.smiles_std == "CC(=O)[O-]"
    assert _formal_charge(r.smiles_std) == -1
