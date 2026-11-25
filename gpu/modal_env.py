"""Make the Modal client work behind this sandbox's TLS-inspecting proxy: trust the proxy CA for gRPC and HTTP.
(Only needed in the cloud sandbox; harmless elsewhere.) Token id/secret come from the environment; the secret is
injected by the sandbox proxy, so MODAL_TOKEN_SECRET may hold a placeholder."""
import os, ssl
CA = "/root/.ccr/ca-bundle.crt"
if os.path.exists(CA):
    import grpclib.client
    def _ctx(self, *a, **k):
        c = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=CA)
        c.set_alpn_protocols(["h2"])
        return c
    grpclib.client.Channel._get_default_ssl_context = _ctx
    os.environ.setdefault("SSL_CERT_FILE", CA)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", CA)
# MODAL_TOKEN_ID must be set in the environment (not stored in the repository)
os.environ.setdefault("MODAL_TOKEN_SECRET", "as-proxyinjectedplaceholder000")
