# Homebrew Formula for Sediment
# Place this in your homebrew-tap repository at Formula/sediment.rb
#
# Users install with:
#   brew tap rendro/tap
#   brew install sediment

class Sediment < Formula
  desc "Semantic memory for AI agents - local-first, MCP-native"
  homepage "https://github.com/rendro/sediment"
  version "0.1.0"

  on_macos do
    on_intel do
      url "https://github.com/rendro/sediment/releases/download/v#{version}/sediment-x86_64-apple-darwin.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256"
    end
    on_arm do
      url "https://github.com/rendro/sediment/releases/download/v#{version}/sediment-aarch64-apple-darwin.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/rendro/sediment/releases/download/v#{version}/sediment-x86_64-unknown-linux-musl.tar.gz"
      sha256 "REPLACE_WITH_ACTUAL_SHA256"
    end
  end

  def install
    bin.install "sediment"
  end

  def caveats
    <<~EOS
      To use Sediment with Claude Desktop or Cursor:

      1. Initialize your project:
           cd your-project && sediment init

      2. Add to your MCP config:
           sediment config

      For Claude Desktop, add to:
        ~/Library/Application Support/Claude/claude_desktop_config.json

      For Cursor, add to your MCP settings.
    EOS
  end

  test do
    assert_match "Semantic memory for AI agents", shell_output("#{bin}/sediment --help")
  end
end
