async function request(path, options) {
  const res = await fetch(`/api${path}`, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export const getStats = () => request("/stats");
export const ask = (question, k, mode) =>
  request("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, k, mode }),
  });
export const upload = (files) => {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  return request("/upload", { method: "POST", body: form });
};
export const removeDoc = (name) => request(`/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
