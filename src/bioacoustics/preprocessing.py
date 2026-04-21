from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

from .data import is_soundscape


def get_labels(df, df_taxonomy):

    class_encoder = MultiLabelBinarizer()
    primary_encoder = MultiLabelBinarizer()

    class_encoder.fit(df_taxonomy["class_name"].apply(lambda x: [x]))
    primary_encoder.fit(df_taxonomy["primary_label"].apply(lambda x: [x]))

    primary_to_class = df_taxonomy.set_index("primary_label")["class_name"]

    # TODO: and secondary labels? - completely ignore them?
    if is_soundscape(df):
        y_class = class_encoder.transform(
            df["primary_label"]
            .apply(lambda x: x.split(";"))
            .apply(
                lambda x: list({primary_to_class[primary_label] for primary_label in x})
            )
        )

        y_primary = primary_encoder.transform(
            df["primary_label"].apply(lambda x: x.split(";"))
        )
    else:
        y_class = class_encoder.transform(df["class_name"].apply(lambda x: [x]))
        y_primary = primary_encoder.transform(df["primary_label"].apply(lambda x: [x]))

    y_class = pd.DataFrame(
        y_class,  # type: ignore
        columns=class_encoder.classes_,
        index=df.index,
    )

    y_primary = pd.DataFrame(
        y_primary,  # type: ignore
        columns=primary_encoder.classes_,
        index=df.index,
    )

    return y_class, y_primary
