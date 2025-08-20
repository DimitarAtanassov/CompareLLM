// ui/app/backend/[...path]/route.ts
const API_ORIGIN = process.env.INTERNAL_API_ORIGIN || "http://api:8080";

export const dynamic = "force-dynamic";
export const fetchCache = "force-no-store";

function upstreamURL(req: Request) {
  const url = new URL(req.url);
  // strip the "/backend/" prefix from the pathname
  const path = url.pathname.replace(/^\/backend\//, "");
  return `${API_ORIGIN}/${path}${url.search}`;
}

function passthroughHeaders(res: Response) {
  const h = new Headers();
  const ct = res.headers.get("content-type");
  if (ct) h.set("content-type", ct);
  h.set("cache-control", "no-store");
  return h;
}

export async function GET(req: Request) {
  const res = await fetch(upstreamURL(req), { cache: "no-store" });
  return new Response(res.body, { status: res.status, headers: passthroughHeaders(res) });
}

export async function POST(req: Request) {
  const res = await fetch(upstreamURL(req), {
    method: "POST",
    headers: { "content-type": req.headers.get("content-type") ?? "application/json" },
    body: await req.arrayBuffer(), // works for JSON + NDJSON streaming
    cache: "no-store",
    redirect: "manual",
  });
  return new Response(res.body, { status: res.status, headers: passthroughHeaders(res) });
}

// (Optional) add other verbs if you need them:
export const PUT = POST;
export const PATCH = POST;
export const DELETE = POST;
