# Local data for M3 (no automated downloads)

No data have been downloaded. Obtain MovieLens-1M from GroupLens according to its
license and extract it yourself. Point --data-dir at the directory containing:

- ratings.dat: UserID::MovieID::Rating::Timestamp (Unix seconds)
- movies.dat: MovieID::Title::Genres (Latin-1; genres separated by |)

For Amazon, supply one small category's review and metadata JSONL files (plain or
.gz). The explicitly supported adapters are:

- 2018: reviews reviewerID, asin, overall, unixReviewTime (seconds); metadata asin,
  title, category/categories, description, brand.
- 2023: reviews user_id, parent_asin, rating, timestamp (milliseconds); metadata
  parent_asin, title, main_category/categories, description, store.

Use --amazon-schema to select the format. Legacy Python-literal records are not
JSONL and are not evaluated by this loader. IDs are mapped from sorted strings;
items.parquet saves source item IDs. Missing metadata yields empty content fields.
Review duplicates are retained as separate events. Ratings >= --min-rating
(default 4) become implicit positive interactions; lower ratings are discarded.

KuaiSAR remains a deliberate stub pending verification against an actual release.
See download_encoder.md for separate manual encoder-weight preparation.
