// The atlas files, fetched once per release and kept for the last few views:
// a decoded figure can be tens of megabytes, so the cache is short.
import { fetchAtlasFile } from '../api/client.js';

const MAX_FILES = 8;
const files = new Map();

export function atlasFile(release, solute, name) {
  const key = `${release}/${solute}/${name}`;
  let p = files.get(key);
  if (p) {
    files.delete(key); // most recently used last
    files.set(key, p);
    return p;
  }
  p = fetchAtlasFile(solute, name);
  p.catch(() => files.delete(key));
  files.set(key, p);
  while (files.size > MAX_FILES) files.delete(files.keys().next().value);
  return p;
}

export function clearAtlasFiles() {
  files.clear();
}
