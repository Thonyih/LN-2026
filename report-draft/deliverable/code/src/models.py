import re, numpy as np
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from config import FIELDS, LABELS

STEM = PorterStemmer()


def stem_tokens(t):
    return [STEM.stem(w) for w in re.findall(r"\b\w\w+\b", t.lower())]


def field(df, name):
    return df[name].tolist()


def joined(df):
    return df[FIELDS].agg(" | ".join, axis=1).tolist()


def model(spec):
    family = spec["family"]
    c = spec.get("C", 1.0)
    weight = "balanced" if spec.get("balanced") else None
    if family == "baseline":
        return Pipeline(
            [
                ("select", FunctionTransformer(field, kw_args={"name": "description"})),
                ("tfidf", TfidfVectorizer(tokenizer=stem_tokens, token_pattern=None)),
                ("clf", MultinomialNB(alpha=1.0)),
            ]
        )
    if family == "wordchar_svm":
        branches = []
        for f in FIELDS if spec["condition"] != "clinical" else FIELDS[:3]:
            for tag, vec in [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=40000,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        sublinear_tf=True,
                        max_features=30000,
                    ),
                ),
            ]:
                branches.append(
                    (
                        f + "_" + tag,
                        Pipeline(
                            [
                                (
                                    "select",
                                    FunctionTransformer(field, kw_args={"name": f}),
                                ),
                                ("tfidf", vec),
                            ]
                        ),
                    )
                )
        return Pipeline(
            [
                ("features", FeatureUnion(branches)),
                (
                    "clf",
                    LinearSVC(
                        C=c, class_weight=weight, max_iter=10000, random_state=42
                    ),
                ),
            ]
        )
    if family in ["word_lr", "lsa"]:
        steps = [
            ("select", FunctionTransformer(joined)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000
                ),
            ),
        ]
        if family == "lsa":
            steps += [
                ("svd", TruncatedSVD(128, random_state=42)),
                ("normalise", Normalizer()),
            ]
        return Pipeline(
            steps
            + [
                (
                    "clf",
                    LogisticRegression(
                        C=c, class_weight=weight, max_iter=3000, random_state=42
                    ),
                )
            ]
        )
    if family == "minilm_lr":
        return LogisticRegression(
            C=c, class_weight=weight, max_iter=3000, random_state=42
        )
    raise ValueError(family)


def score(est, x):
    s = (
        est.predict_proba(x)
        if hasattr(est, "predict_proba")
        else est.decision_function(x)
    )
    return np.asarray(s)[:, [list(est.classes_).index(l) for l in LABELS]]
