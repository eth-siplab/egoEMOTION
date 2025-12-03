import pandas as pd


def load_personality_data(participants, data_path, personality_domains):
    # Load personality questionnaire data and convert to DataFrame with numbers
    raw = pd.read_csv(f'{data_path}/personality_questionnaire_results.csv', sep=";", header=None, dtype=str)
    ids = raw.iloc[1, 1:].tolist()
    scales = raw.iloc[2:, 0].str.strip()
    mat = (raw.iloc[2:, 1:]
           .map(lambda x: str(x).replace(",", "."))  # comma → dot
           .apply(pd.to_numeric, errors="coerce")
           .set_axis(scales, axis=0)
           .set_axis(ids, axis=1))

    # Remove the first two rows and keep only participants wanted
    df_all_full = mat.T.copy()  # rows = participants, cols = all rows in the CSV
    df_all_full.columns.name = None  # remove the printed “0” column‑name label
    df_all_full = df_all_full.loc[participants]
    dup_mask = df_all_full.columns.duplicated(keep='first')

    # Split into means and T-scores and remove empty columns
    df_means = df_all_full.loc[:, ~dup_mask]  # all mean rows
    df_tscores = df_all_full.loc[:, dup_mask]  # all T‑score rows
    df_means = df_means.dropna(axis=1, how='all')
    df_tscores = df_tscores.dropna(axis=1, how='all')

    # Only keep desired personality domains
    df_means = df_means[personality_domains]
    df_tscores = df_tscores[personality_domains]

    return df_means, df_tscores


def load_affective_data(participants, data_path, session):
    dfs = []
    for p in participants:
        df = pd.read_csv(f"{data_path}/{p}/Session_{session}_{p}.csv")
        df['Valence'] = df['Valence'].astype(float) - 4  # Recenter valence
        df = df.drop(columns=['Session Start', 'Video Start', 'Q1 Start', 'Q2 Start', 'Q2 End', 'Session End'])
        df.insert(0, 'participant', p)  # Insert the participant column at position 0
        # Take mean of multiple same activity names
        if session == 'B':
            df = df.groupby(['participant', 'Activity Index', 'Activity Name', 'Activity ID'],
                            sort=False, as_index=False).mean()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)



def get_df_labels(dfs_all, participants, session_label, n_chunks_all=None):
    df_parts = []
    for p in participants:
        if session_label in ('A', 'B'):
            dfp = dfs_all[session_label][dfs_all[session_label]['participant'] == p].copy()
        else:
            dfA = dfs_all['A'][dfs_all['A']['participant'] == p]
            dfB = dfs_all['B'][dfs_all['B']['participant'] == p]
            dfp = pd.concat([dfA, dfB], ignore_index=True)

        if n_chunks_all is not None:
            counts = n_chunks_all.get(p)
            if counts is None:
                raise KeyError(f"No n_chunks_all entry for participant {p}")
            if len(counts) != len(dfp):
                raise ValueError(
                    f"Participant {p}: expected {len(dfp)} tasks, "
                    f"but got counts array of length {len(counts)}"
                )
            dfp = dfp.loc[dfp.index.repeat(counts)].reset_index(drop=True)

        df_parts.append(dfp)
    df_all_labels = pd.concat(df_parts, ignore_index=True)
    return df_all_labels
