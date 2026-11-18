# Measured Data tab

The web app's **Measured Data** tab shows the curated experimental database of water properties (liquid,
phase-equilibrium, amorphous, kinetic, dynamics and structure tables) next to the waterEoS models. The database itself
is compiled outside this repository; this repository holds the loader, the read-only API and the React views.

## How the pieces fit

```
fitting_data/  (CSV tables + sources.csv, kept outside the repository)
     │  python -m watereos_api.measured.load --source <fitting_data> --db <url>
     ▼
SQL database   (SQLite file locally; PostgreSQL on Railway)
     │  watereos_api/routes/measured.py   GET /api/data/...   (read-only, cached per release)
     ▼
React          watereos-web/src/screens/MeasuredData.jsx (figures built in the browser)
```

* `watereos_api/measured/schema.py`: published columns, table definitions, labels, and which model attribute
  is drawn over which property.
* `watereos_api/measured/load.py`: the loader (see its docstring).
* `watereos_api/measured/store.py`: database access and the per-release response cache.
* `watereos_api/measured/curves.py`: model curves and surfaces from the registry; IAPWS R14-08 melting curves,
  the Holten et al. (2014) T_H line and the Murphy & Koop (2005) vapor pressures.

The API reads the database named by `WATEREOS_DATA_URL` (or `DATABASE_URL`). Without one, the tab says the data are not
configured, and the rest of the app is unaffected.

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/data/release` | Release id and counts, tables and properties (unit, locus, log axis, overlay models), notice and citation |
| `GET /api/data/points?table=&property=` | Plotting columns for one table and property (all rows when both are omitted), as columns |
| `GET /api/data/rows/{id}` | Every published column of one row, with its source |
| `GET /api/data/sources` | The cited studies |
| `GET /api/data/download` | The whole release as a zip: `measured_water_data.csv`, `sources.csv`, `README.txt` |
| `GET /api/data/lines` | Melting, homogeneous-nucleation and vapor-pressure lines |
| `POST /api/data/model-curves` | Model isobars or isotherms over a property (at most 12 models × 16 curves) |
| `POST /api/data/model-surface` | A model surface over T and P for the 3D view |

Responses are computed once per release and served with an `ETag`; the API re-reads the release id at most once a
minute, so a reload of the database goes live without a restart. Responses are gzip-compressed. Nothing needs sign-in.

Model overlays are offered only for the liquid and dynamics properties that a model computes, and only for equations of
real water: the registry's fits to simulated model waters (`is_model_water`) and the vapor model are left out.

## What is published

Every table and every source of the database, with these changes made by the loader:

* Private paths are removed. The `via` column, which names the file the value was read from on the build machine,
  becomes the published supplement it points to (for example "Holten, Sengers & Anisimov (2014) supplementary material:
  Data tables/rho-Kell-Whalley-1975.txt"), or nothing for the database's own transcriptions. Mentions of internal files
  in the notes are reworded. `sources.csv`'s local PDF paths, catalog status and reading-note flags are dropped. A final
  scan refuses to load if any local path is left; add a rule to `load.py` when it does. `--forbid TERM` (repeatable)
  adds names of your own, such as local folder names, that must never appear; keep those on the command line rather
  than in the code, which is public.
* Added: `u_std` (absolute standard uncertainty, k = 1, as `fitting_data/scripts/fdata.py` computes it), the rate in
  K/min, the publication year, and a release record.

The views hide D2O and the `quoted` and `correlation` rows by default; the filters bring them back. Kinetic, dynamics
and structure tables carry a "not thermodynamic data" banner.

The site and the download carry this notice (`schema.NOTICE`): the values come from the cited published studies;
waterEoS asserts no rights over them and attaches no license; values from supplementary files stay under each
publisher's terms; cite the original studies and cite waterEoS for the compilation.

## Local work

```bash
python -m watereos_api.measured.load --source /path/to/fitting_data          # writes .measured-data/measured.sqlite
WATEREOS_DATA_URL=sqlite:///.measured-data/measured.sqlite python -m uvicorn watereos_api.app:app --port 8000
cd watereos-web && npm run dev                                                 # http://localhost:5173, proxies /api
```

`--dry-run` reads and checks the tables without writing. `.measured-data/` is ignored by git and Docker.

Tests: `python -m pytest tests/test_api_measured.py` (a made-up miniature database, no real values) and
`npx vitest run` in `watereos-web/`. With Node 25 or later, run vitest with `NODE_OPTIONS=--no-experimental-webstorage`:
Node's own `localStorage` otherwise hides jsdom's and every frontend test fails in `tests/setup.js`.

## Production on Railway

The Railway service is **not** connected to GitHub: merging to `master` deploys nothing. Changes to the production
project are the owner's decision.

### The database

A private PostgreSQL service, `Postgres`, in the visualizer's Railway project. It has no public address. The
visualizer service reads it through Railway's private network:

```
WATEREOS_DATA_URL=${{Postgres.DATABASE_URL}}
```

The database sorts text with `en_US.utf8`; the API sorts rows itself, so its answers and the download are the same as
from a local SQLite copy.

### Loading a release

Load when a build of `fitting_data` is worth publishing, not after every build. The load replaces the previous release
in one transaction, and the site shows the new release id within a minute, with no redeploy.

The database is reached through Railway's SSH with a local port forward. This needs, once, an SSH key on the machine
registered with the Railway account that owns the project:

```bash
ssh-keygen -t ed25519
```

```bash
railway ssh keys add
```

Then, from a folder linked to the project (`railway link`):

```bash
cfg=$(mktemp)
railway ssh config --service Postgres --path "$cfg" --alias watereos-pg --identity-file ~/.ssh/id_ed25519
ssh -F "$cfg" -o UserKnownHostsFile="$cfg.hosts" -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes \
    -f -N -L 127.0.0.1:15432:localhost:5432 watereos-pg
railway run --service Postgres -- sh -c 'cd /path/to/waterEoS && \
    WATEREOS_DATA_URL="postgresql://$PGUSER:$PGPASSWORD@127.0.0.1:15432/$PGDATABASE" \
    python -m watereos_api.measured.load --source /path/to/fitting_data'
pkill -f 'L 127.0.0.1:15432:localhost:5432 watereos-pg'
```

The SSH settings and host key go to temporary files, so `~/.ssh/config` is not touched. `railway run` hands the
database password to the loader; it is never typed and does not appear on the command line or in the shell history.
Add `--forbid` terms (local folder names) to the loader as needed. To check a load, point the API at the same URL
and compare `/api/data/release` with a local SQLite load of the same tables: the release id is the date and a hash of
the inputs, so the two match.

The alternative, a public TCP proxy on the database (`railway tcp-proxy create --service Postgres --port 5432`), makes
it reachable from the internet with password authentication. It is not used.

### Deploying the app

From a clean copy of the merged commit, so that only tracked files are uploaded:

```bash
d=$(mktemp -d) && git archive origin/master | tar -x -C "$d" && cd "$d" && \
  railway up --detach --project <project id> --service <service id> --environment <environment id>
```

The ids are in the service's dashboard URL. The image installs `psycopg[binary]` from `requirements-web.txt`.
