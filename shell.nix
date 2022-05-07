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

  pkgs_1703 = import (builtins.fetchGit {
    # Descriptive name to make the store path easier to identify
    name = "nixos-1703";
    url = "https://github.com/nixos/nixpkgs/";
    # Commit hash for nixos-unstable as of 2018-09-12
    # `git ls-remote https://github.com/nixos/nixpkgs nixos-unstable`
    #allRefs = true;
    ref = "release-17.09";
    rev = "3ba3d8d8cbec36605095d3a30ff6b82902af289c";
    #rev = "1849e695b00a54cda86cb75202240d949c10c7ce"; 1703
    #rev = "a7ecde854aee5c4c7cd6177f54a99d2c1ff28a31"; 2111
  }) { };

  python-with-my-packages = (pkgs_1703.python27.withPackages (p: with p; [
    cython
    numpy
    scipy
    scikitimage
    matplotlib
    ipython
    h5py
    #leveldb
    networkx
    nose
    pandas
    #python-dateutil
    pyparsing
    dateutil
    protobuf
    gflags
    pyyaml
    pillow
    six
  ])).override (args: { ignoreCollisions = true; });

  myAppEnv = pkgs_new.poetry2nix.mkPoetryEnv {
    projectDir = ./.;
    python = pkgs_import.python3;
  };
in python-with-my-packages
  #{ inherit myAppEnv pkgs_new pkgs_import;}
#pkgs.mkShell {
#  buildInputs = [
#    python-with-my-packages
#    # other dependencies
#  ];
#  shellHook = ''
#    PYTHONPATH=${python-with-my-packages}/${python-with-my-packages.sitePackages}
#    # maybe set more env-vars
#  '';
#}
