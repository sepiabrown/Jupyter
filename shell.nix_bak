{ pkgs_import ? import ../nixpkgs {} }:
let
  pkgs_new = import ../nixpkgs {
    overlays = [
      (self: super: {
        #poetry2nix = super.poetry2nix.override { pkgs = pkgs_import;};
        #poetry2nix = super.poetry2nix.overrideScope' (p2nixself: p2nixsuper: {
        #  defaultPoetryOverrides = p2nixsuper.defaultPoetryOverrides.extend (pyself: pysuper: {
        #    pillow = super.python3.pkgs.pillow;#.overridePythonAttrs ( old: {
        #    #  nativeBuildInputs = [ pkgs.buildPackages.pkg-config self.pytest-runner ] ++ (old.nativeBuildInputs or [ ]);
        #    #  buildInputs = with pkgs; [ freetype libjpeg zlib libtiff libwebp tcl lcms2 ] ++ (old.buildInputs or [ ]);
        #    #}); 
        #    #import ../nixpkgs/pkgs/development/python-modules/pillow {inherit self pyself;};
        #    #{ inherit (pyself) pillow; };#(import ./pillow/default.nix) {}; #default.nix { inherit (pysuper) pillow; };
        #  });
        #});
      })
    ];
  };
  myAppEnv = pkgs_new.poetry2nix.mkPoetryEnv {
    projectDir = ./.;
    python = pkgs_import.python3;
  };
in #myAppEnv.env
  { inherit myAppEnv pkgs_new pkgs_import;}
