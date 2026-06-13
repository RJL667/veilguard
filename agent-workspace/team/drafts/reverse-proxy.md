# Reverse Proxies

## Overview

A reverse proxy is a server that sits between external clients and one or more backend servers, intercepting incoming requests and forwarding them on behalf of those backends. Unlike a forward proxy (which acts on behalf of clients), a reverse proxy acts on behalf of servers — the client communicates only with the proxy and has no direct visibility into the backend infrastructure. Reverse proxies can terminate TLS, cache responses, compress traffic, and enforce authentication before requests ever reach an application server. Common implementations include Nginx, HAProxy, and Cloudflare's edge network.

## Use Cases

**Load Balancing:** A reverse proxy distributes incoming traffic across a pool of backend servers, preventing any single instance from becoming a bottleneck. If one server fails, the proxy routes requests to healthy nodes, improving both availability and fault tolerance.

**TLS Termination:** Rather than configuring TLS on every backend service, a reverse proxy handles certificate management and decryption at the edge. Backend servers receive plain HTTP internally, simplifying certificate rotation and reducing per-service overhead.
