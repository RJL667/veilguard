# CI Free-Tier Comparison: GitHub Actions vs GitLab CI

_Researched: 2024_

---

## Method

Web searches targeting official documentation and pricing pages for both platforms: GitHub Docs (Actions billing, plan limits) and GitLab Docs (compute minutes, storage quotas, pricing page).

---

## Findings

| Dimension | GitHub Actions (Free) | GitLab CI (Free / SaaS) |
|---|---|---|
| Compute minutes/month | 2,000 min | 400 min |
| Artifact/repo storage | 500 MB | 10 GiB per project (repo + LFS) |
| Public repos | Unlimited, free | Unlimited, free |
| Self-hosted runners | Free (usage not counted against quota) | Free (usage not counted against quota) |
| Overage | Pay-as-you-go | Purchase additional minutes |

GitHub Actions provides **5× more hosted compute minutes** (2,000 vs 400) on the free tier. GitLab's storage allowance is substantially more generous (10 GiB project storage vs 500 MB artifact storage), though the two metrics are not directly comparable.

---

## Recommendation

For teams primarily using GitHub-hosted runners, **GitHub Actions** is the stronger free-tier choice due to its 5× compute minute advantage. Teams already on GitLab, or those needing large artifact/LFS storage, may find GitLab's free tier adequate — especially if they can supply self-hosted runners to bypass the 400-minute cap entirely.
