"""Check recallable flag distribution."""
import lancedb, time

df = lancedb.connect("/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db").open_table("archive").to_pandas()
df = df[df["created_ts"] >= time.time() - 86400]

print("=== recallable distribution by block_class (today) ===")
g = df.groupby(["block_class", "recallable"]).size().reset_index(name="count")
print(g.to_string(index=False))

print()
rec_f = (df["recallable"] == False).sum()
rec_t = (df["recallable"] == True).sum()
print(f"recallable=False: {rec_f}/{len(df)}")
print(f"recallable=True:  {rec_t}/{len(df)}")
