"""Opt-in plural folding in lexical search keys."""

# Standard Library
from dataclasses import replace

# Third Party Library
import pytest

# Package Library
from kgfegmcp.bootstrap import AppState
from kgfegmcp.catalog.models import CatalogPackageRuntime
from kgfegmcp.search.lexical import LexicalIndex
from kgfegmcp.search.models import TextMatchMode, TextOperator, TokenTextMatch
from kgfegmcp.search.normalizers import (
    LEXICAL_NORMALIZER_VERSION,
    LEXICAL_PLURAL_FOLD_NORMALIZER_VERSION,
    fold_plural,
    lexical_normalizer_version,
    normalize_lexical_text,
)

CAPS = "south-africa-dbe-caps-english-home-language-grades-r-3"
RWANDA = "rwanda-reb-mathematics-lower-primary-1-3"
# index_sha256 values from origin/main 2.0.0 (45f07c4). With folding off they must not
# move; update them only when the package data itself changes.
MAIN_INDEX_SHA256 = {
    CAPS: "sha256:f4a7d9df2e80308404af4a35dd3d340d9ebb2831a53b60f436428fb1745682b3",
    RWANDA: "sha256:7bda6b66834405e0622f40a1b3b2e59aa1282c54fb4afb3468b75586aee2b693",
}
ANY_TOKEN = TokenTextMatch(match_mode=TextMatchMode.TOKENS, operator=TextOperator.ANY)


@pytest.mark.parametrize(
    ("plural", "singular"),
    [
        ("pictographs", "pictograph"),
        ("equations", "equation"),
        ("fractions", "fraction"),
        ("faces", "face"),
        ("sizes", "size"),
        ("classes", "class"),
        ("boxes", "box"),
        ("matches", "match"),
        ("wishes", "wish"),
        ("studies", "study"),
        ("quantities", "quantity"),
        ("caches", "cache"),
        ("headaches", "headache"),
        ("axes", "axis"),
        ("halves", "half"),
        ("vertices", "vertex"),
        ("children", "child"),
    ],
)
def test_fold_plural_joins_singular_and_plural(plural: str, singular: str) -> None:
    """A plural and its singular fold to one key."""
    assert fold_plural(plural) == fold_plural(singular)


@pytest.mark.parametrize(
    ("word", "other"),
    [("times", "time"), ("news", "new"), ("series", "sery")],
)
def test_fold_plural_keeps_unrelated_words_apart(word: str, other: str) -> None:
    """Words that only look like plurals keep their own key."""
    assert fold_plural(word) == word
    assert fold_plural(word) != fold_plural(other)


@pytest.mark.parametrize(
    "token",
    [
        "class",
        "bonus",
        "analysis",
        "add",
        "is",
        "bus",
        "tie",
        "12",
        "3s",
        "x",
        "b\u00fccherregals",
        "\u0645\u062f\u0627\u0631\u0633",
    ],
)
def test_fold_plural_leaves_other_tokens(token: str) -> None:
    """Short, numeric, non-ASCII and -ss/-us/-is tokens are unchanged."""
    assert fold_plural(token) == token


def test_folding_is_off_unless_requested() -> None:
    """The default tokenizer is the unfolded one; folding applies token by token."""
    text = "Draw 3 PICTOGRAPHS, then Equations"

    assert normalize_lexical_text(text) == (
        "draw",
        "3",
        "pictographs",
        "then",
        "equations",
    )
    assert normalize_lexical_text(text, fold_plurals=True) == (
        "draw",
        "3",
        "pictograph",
        "then",
        "equation",
    )


def test_normalizer_version_follows_the_setting() -> None:
    """Only an opted-in package reports the folding version."""
    assert lexical_normalizer_version(fold_plurals=False) == LEXICAL_NORMALIZER_VERSION
    assert (
        lexical_normalizer_version(fold_plurals=True)
        == LEXICAL_PLURAL_FOLD_NORMALIZER_VERSION
    )
    assert LEXICAL_NORMALIZER_VERSION == "lexical_nfkc_casefold_tokens_v1"


def test_shipped_packages_keep_their_origin_main_index(
    accepted_state: AppState,
) -> None:
    """No shipped profile opts in, so index versions and hashes equal origin/main."""
    packages = accepted_state.search_service.metadata.packages

    assert {str(m.lexical_normalizer_version) for m in packages} == {
        LEXICAL_NORMALIZER_VERSION
    }
    hashes = {
        str(m.package_identity.framework_id): str(m.index_sha256) for m in packages
    }
    for framework_id, expected in MAIN_INDEX_SHA256.items():
        assert hashes[framework_id] == expected


def _caps_runtime(state: AppState, *, fold: bool) -> CatalogPackageRuntime:
    """Return the CAPS runtime with its profile's plural folding set as given."""
    runtime = next(
        r
        for r in state.catalog_load_result.package_runtimes
        if r.catalog_package.package_identity.framework_id == CAPS
    )
    package = runtime.loaded_package
    policy = package.profile.language_policy.model_copy(
        update={"fold_lexical_plurals": fold}
    )
    profile = package.profile.model_copy(update={"language_policy": policy})

    return replace(
        runtime, loaded_package=package.model_copy(update={"profile": profile})
    )


def _hit_ids(index: LexicalIndex, query: str) -> set[str]:  # type: ignore[type-arg]
    """Return the node IDs a token query finds in one lexical index."""
    return {
        str(candidate.node.node_id)
        for candidate in index.candidates(match=ANY_TOKEN, query=query)
    }


def test_profile_opt_in_makes_singular_and_plural_agree(
    accepted_state: AppState,
) -> None:
    """With the profile flag on, a word and its plural find the same standards."""
    folded = LexicalIndex.from_runtime(_caps_runtime(accepted_state, fold=True))
    exact = LexicalIndex.from_runtime(_caps_runtime(accepted_state, fold=False))

    assert folded.fold_plurals and not exact.fold_plurals
    assert _hit_ids(folded, "sounds")
    assert _hit_ids(folded, "sound") == _hit_ids(folded, "sounds")
    assert _hit_ids(exact, "sound") != _hit_ids(exact, "sounds")
