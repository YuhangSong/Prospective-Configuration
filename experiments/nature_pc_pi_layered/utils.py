import seaborn as sns
from analysis_utils import nature_relplot
import matplotlib


def process_analysis_df(df):
    """Pre-process analysis_df.
    """

    df.insert(
        1,
        'Update weights to',
        df.apply(
            lambda row: learning_duration_to_w(
                row['learning_duration']
            ),
            axis=1,
        )
    )

    return df


def learning_duration_to_w(x):
    return {
        1: r"$w'$",
        1000: r"$w^*$",
    }[x]
