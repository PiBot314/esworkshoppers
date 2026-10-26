# IIIT-H oneM2M portal: access notes

## Working link

http://onem2m.iiit.ac.in:443/webpage/welcome/index.html?context=/~&cseId=in-cse

Login: `iiith_guest` / `iiith_guest`

Use `http://`, not `https://`. The server is plain HTTP on port 443 (Jetty 8, Eclipse OM2M).

## Symptom: "400 Bad Request" on every click

Opening the page without the query string (`.../index.html`) lets you log in, but every
resource you click returns `400 ... Malformed URI`.

**Cause:** the page's script (`om2m.js`) reads `context` and `cseId` from the URL's query
string and puts them in front of every request. Without them, both are empty:

- Login sends `GET /?rcn=5&lvl=1`. This server answers that with the CSE root, so login
  appears to work.
- The tree's child links are paths like `/in-cse/CAE745300044`. Without the `/~` context
  prefix, the server rejects them with 400.

Verified with curl on 2026-10-15:

| Request | Result |
|---|---|
| `GET /in-cse/acp-931819749` (no context) | 400 Malformed URI |
| `GET /~/in-cse/CAE745300044` | 200 OK |
| `GET /~/in-cse/acp-...` | 403 Forbidden |

The 403 on `acp-...` items is expected: those are access-control policies, and the guest
account can't read them. The `CAE...` items (devices) and their data load fine.

## Direct API access (curl)

OM2M takes credentials in the `X-M2M-Origin` header as `user:password`, not as Basic auth:

```bash
curl -H 'X-M2M-Origin: iiith_guest:iiith_guest' -H 'Accept: application/json' \
     'http://onem2m.iiit.ac.in:443/~/in-cse?rcn=5&lvl=1'
```

Child resources go under `/~` followed by the child's `val`, e.g. `/~/in-cse/CAE745300044`.
