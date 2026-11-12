{
  description = "llm-inference-book - Learning LLM Inference the Hard Way, built with mdBook";

  inputs = {
    # nixos-unstable is required for mdBook >= 0.5: 0.5 renamed the book.toml key
    # `curly-quotes` to `smart-punctuation`, and nixpkgs 24.05/24.11 ship mdBook 0.4.x,
    # which refuses to deserialize a 0.5-era book.toml.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));
    in
    {
      formatter = forAllSystems (pkgs: pkgs.nixfmt-rfc-style);

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.mdbook
            pkgs.git
            pkgs.ripgrep
            pkgs.fd
            pkgs.jq
            pkgs.python3
          ];

          shellHook = ''
            echo "dev shell - mdbook $(mdbook --version | sed 's/^mdbook v//') ($(uname -m) $(uname -s))"
            echo "  mdbook serve --open   # live preview at http://localhost:3000"
            echo "  mdbook build          # writes the site to book/"
            echo "  bash scripts/verify.sh  # the gate CI runs"
            echo "  nix build             # reproducible build; result/ is the whole site"
          '';
        };
      });

      packages = forAllSystems (
        pkgs:
        let
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter =
              path: type:
              let
                base = builtins.baseNameOf (toString path);
              in
              !(builtins.elem base [
                "book"
                "result"
                ".direnv"
                ".git"
                ".DS_Store"
              ]);
          };
        in
        {
          default = pkgs.stdenvNoCC.mkDerivation {
            pname = "llm-inference-book";
            version = "unstable";
            inherit src;
            nativeBuildInputs = [ pkgs.mdbook ];
            buildPhase = ''
              runHook preBuild
              mdbook build --dest-dir "$out"
              runHook postBuild
            '';
            dontInstall = true;
          };
        }
      );
    };
}
