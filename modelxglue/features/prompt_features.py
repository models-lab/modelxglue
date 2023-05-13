
def get_prompt_features(df, split='train'):
    context_cols = [c for c in df.columns if c not in ['ids', 'target']]
    corpus = []
    pass
