"""Literature references of the models (watereos.references)."""
import re

from watereos.model_registry import MODEL_ORDER
from watereos.references import MODEL_REFERENCES, REFERENCE_GROUPS, REFERENCES

DOI = re.compile(r'^10\.\d{4,9}/\S+$')


def test_every_model_has_references():
    assert set(MODEL_REFERENCES) == set(MODEL_ORDER)
    for model, keys in MODEL_REFERENCES.items():
        assert keys, model
        assert all(key in REFERENCES for key in keys), model


def test_every_reference_has_a_doi_and_one_group():
    grouped = [key for _, keys in REFERENCE_GROUPS for key in keys]
    assert sorted(grouped) == sorted(REFERENCES)
    for key, ref in REFERENCES.items():
        assert ref['short'] and ref['citation'], key
        if key == 'bridge29':
            assert ref['doi'] is None and 'Manuscript' in ref['citation']
        else:
            assert DOI.match(ref['doi']), key
        for note in ref['notes']:
            assert note['text'] and DOI.match(note['doi']), key
