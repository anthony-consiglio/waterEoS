# H₂O Phase Diagram tab (water + NaCl)

The tab is the atlas of [cryo-isochoric](#where-the-figures-come-from) for water + sodium chloride, in that atlas's
own light style: the phase equilibria of water and its NaCl solutions in pressure, temperature, specific volume and
composition, drawn as sections, surfaces, the equilibrium complex in 3-D and the convex-hull construction it comes
from. At zero NaCl the diagrams are those of pure water; the tab opens on the isoplethal T–p diagram at x = 0, the
p–T diagram of water. The atlas's process view (the isochoric chamber) is not part of the tab.

Range: T above 229.2 K (the liquid–liquid critical point of the Holten 2014 liquid), 1 Pa to 400 MPa, ices Ih, II,
III, V and VI, vapor, and NaCl up to 22 wt % (the fitted range of the solution model, 4.83 mol/kg). Composition is
shown in wt % (what is computed), mol %, mol/kg or vol %.

## How the pieces fit

```
cryo-isochoric-gallery build nacl --out DIR     (private package, on the owner's machine)
        │  DIR/index.html (manifest: views, captions, color key)
        │  DIR/data/nacl/<view>.js, info.js
        ▼
python -m watereos_api.atlas.load --source DIR  (watereos_api/atlas/load.py)
        │  checks, gzip, one transaction
        ▼
atlas_files, atlas_release                      (the measured-data database: SQLite locally, PostgreSQL on Railway)
        │
        ▼
GET /api/atlas/...                              (watereos_api/routes/atlas.py; serves the stored files as they are)
        │
        ▼
H₂O Phase Diagram tab                           (watereos-web/src/screens/H2OPhaseDiagram.jsx, src/h2oAtlas/)
```

The API computes nothing: the figures are Plotly JSON built by cryo-isochoric, stored gzip-compressed and sent as
stored to clients that accept gzip. The page converts them to other composition units itself (`src/h2oAtlas/atlas.js`,
ported from the atlas page), so one stored figure serves every unit.

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/atlas/manifest` | `configured`, `release_id`, `created_at`, `solutes`, `groups`, `views` (title, caption, log switch), `key` (colors), `quality`, `built_at`, `builds` |
| `GET /api/atlas/{solute}/{name}` | one file: a figure (`d-T-xv`, its logarithmic twin `d-T-xv-log`, `s3-x`, `proj-pvT`, `hull`, ...) or the system notes (`info`) |

Files carry an ETag (the SHA-256 of their content) and `Cache-Control: public, max-age=300`. Without a database the
manifest answers `{"configured": false}` and the tab says the phase diagrams are not loaded.

## Where the figures come from

cryo-isochoric (a private package, hence the precomputed figures) computes the equilibria as the lower
convex hull of the Helmholtz energy over specific volume and composition, one polytope per temperature: the liquid of
Holten et al. (2014) and the SeaFreeze ices from waterEoS, the waterEoS vapor, and the osmotic virial solution model of
cryo-osmo (private), with solute properties from the cryo-toolbox molecule database (private). Each figure's
notes in the tab (Model, Invariant points) give the model and its coefficients.

## Local work

Build the atlas in the cryo-isochoric checkout (about 10 s; 52 MB, 20 MB compressed), then load it:

```bash
cd /path/to/cryo-isochoric && .venv/bin/cryo-isochoric-gallery build nacl --out /tmp/nacl-atlas
```

```bash
python -m watereos_api.atlas.load --source /tmp/nacl-atlas
```

```bash
WATEREOS_ATLAS_URL=sqlite:///.atlas-data/atlas.sqlite uvicorn watereos_api.app:app --port 8000
```

The loader's default database is `$WATEREOS_ATLAS_URL`, `$WATEREOS_DATA_URL`, `$DATABASE_URL`, then
`.atlas-data/atlas.sqlite` (git-ignored); the API reads the same variables in that order. Keep the local atlas in its
own SQLite file: the measured-data loader replaces its whole SQLite file and would drop the atlas tables. In PostgreSQL
each loader replaces only its own tables, so both share the production database. `--dry-run` checks without writing;
`--forbid TERM` refuses to load a file whose text contains TERM.

The loader refuses a build with a missing view or logarithmic twin, a file that is not the figure it is named for, a
value that is not JSON (NaN, Infinity), or a local path in any published text (base64 array data is not scanned).

## Production on Railway

The atlas tables live in the visualizer's private `Postgres` service, next to the measured data; the service already
reads that database (`WATEREOS_DATA_URL=${{Postgres.DATABASE_URL}}`), so nothing is configured for the tab. Load the
atlas through Railway's SSH with a local port forward, as for the measured data (docs/measured_data.md, "Loading a
release"), from a folder linked to the project:

```bash
cfg=$(mktemp)
railway ssh config --service Postgres --path "$cfg" --alias watereos-pg --identity-file ~/.ssh/id_ed25519
ssh -F "$cfg" -o UserKnownHostsFile="$cfg.hosts" -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes \
    -f -N -L 127.0.0.1:15432:localhost:5432 watereos-pg
railway run --service Postgres -- sh -c 'cd /path/to/waterEoS && \
    WATEREOS_ATLAS_URL="postgresql://$PGUSER:$PGPASSWORD@127.0.0.1:15432/$PGDATABASE" \
    python -m watereos_api.atlas.load --source /tmp/nacl-atlas'
pkill -f 'L 127.0.0.1:15432:localhost:5432 watereos-pg'
```

The load replaces the previous atlas in one transaction and touches no measured-data table; the site serves the new
release within a minute, with no redeploy. Deploy the app itself as described in docs/measured_data.md ("Deploying
the app"). Each load to production is the owner's decision.
