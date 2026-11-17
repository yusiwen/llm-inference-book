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
            # The gate needs only the standard library; the *instruments* need four packages, and a bare
            # `pkgs.python3` is how `nix develop --command python3 scripts/fuse.py` came to fail with
            # `ModuleNotFoundError: No module named 'regex'` while `verify.sh` stayed green - the instruments
            # are not part of the gate, so nothing noticed. Versions are pinned by flake.lock, which is the
            # point: an instrument whose numpy can change under it cannot back a number that claims a commit.
            (pkgs.python3.withPackages (ps: [
              ps.regex
              ps.jinja2
              ps.numpy
              ps.tokenizers
            ]))
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
