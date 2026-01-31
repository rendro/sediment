#!/usr/bin/env bash
set -euo pipefail

REPO="rendro/sediment"
INSTALL_DIR="${SEDIMENT_INSTALL_DIR:-/usr/local/bin}"

# Detect OS
OS="$(uname -s)"
case "$OS" in
  Darwin) os="apple-darwin" ;;
  Linux)  os="unknown-linux-gnu" ;;
  *) echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

# Detect architecture
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) arch="x86_64" ;;
  arm64|aarch64) arch="aarch64" ;;
  *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

TARGET="${arch}-${os}"

# Linux aarch64 not yet supported
if [ "$os" = "unknown-linux-gnu" ] && [ "$arch" = "aarch64" ]; then
  echo "Linux aarch64 binaries are not yet available." >&2
  echo "Install from source: cargo install sediment" >&2
  exit 1
fi

# Get latest version
echo "Fetching latest release..."
VERSION="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')"
if [ -z "$VERSION" ]; then
  echo "Failed to determine latest version" >&2
  exit 1
fi
echo "Installing sediment v${VERSION} (${TARGET})..."

# Download and extract
TARBALL="sediment-${TARGET}.tar.gz"
URL="https://github.com/${REPO}/releases/download/v${VERSION}/${TARBALL}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

curl -fsSL "$URL" -o "${TMPDIR}/${TARBALL}"

# Verify checksum if available
CHECKSUM_URL="https://github.com/${REPO}/releases/download/v${VERSION}/checksums.txt"
if curl -fsSL "$CHECKSUM_URL" -o "${TMPDIR}/checksums.txt" 2>/dev/null; then
  echo "Verifying checksum..."
  EXPECTED="$(grep "${TARBALL}" "${TMPDIR}/checksums.txt" | awk '{print $1}')"
  if [ -n "$EXPECTED" ]; then
    if command -v sha256sum &>/dev/null; then
      ACTUAL="$(sha256sum "${TMPDIR}/${TARBALL}" | awk '{print $1}')"
    elif command -v shasum &>/dev/null; then
      ACTUAL="$(shasum -a 256 "${TMPDIR}/${TARBALL}" | awk '{print $1}')"
    else
      echo "Warning: no sha256sum or shasum found, skipping verification" >&2
      ACTUAL="$EXPECTED"
    fi
    if [ "$ACTUAL" != "$EXPECTED" ]; then
      echo "Checksum verification failed!" >&2
      echo "  Expected: $EXPECTED" >&2
      echo "  Actual:   $ACTUAL" >&2
      exit 1
    fi
    echo "Checksum verified."
  else
    echo "Warning: no checksum found for ${TARBALL} in checksums.txt" >&2
  fi
else
  echo "Warning: checksums.txt not available, skipping verification" >&2
fi

tar -xzf "${TMPDIR}/${TARBALL}" -C "$TMPDIR"

# Install
if [ -w "$INSTALL_DIR" ]; then
  mv "${TMPDIR}/sediment" "${INSTALL_DIR}/sediment"
else
  echo "Installing to ${INSTALL_DIR} (requires sudo)..."
  sudo mv "${TMPDIR}/sediment" "${INSTALL_DIR}/sediment"
fi

chmod +x "${INSTALL_DIR}/sediment"

echo "Installed sediment to ${INSTALL_DIR}/sediment"

# Check PATH
if ! command -v sediment &>/dev/null; then
  echo ""
  echo "Add ${INSTALL_DIR} to your PATH:"
  echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
fi
