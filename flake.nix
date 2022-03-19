{
  description = "MS thesis environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; # poetry doesn't work at nixos-20.09
    jupyterWith.url = "github:tweag/jupyterWith";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, jupyterWith, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "x86_64-darwin" ] (system:
      let
        pkgs = import nixpkgs {
          system = system;
          overlays = (builtins.attrValues jupyterWith.overlays); # ++ [ (import ./haskell-overlay.nix) ];
        };

        python = pkgs.poetry2nix.mkPoetryEnv {
          projectDir = ./my-python-package;
          extraPackages = ps : [ ps.pip ];
          editablePackageSources = {
              ronald = ./my-python-package/ronald_bdl/ronald_bdl;
          };
          # poetrylock = ./my-python-package/poetry.lock;
        };

        pyproject = builtins.fromTOML (builtins.readFile ./my-python-package/pyproject.toml);
        depNames = builtins.attrNames pyproject.tool.poetry.dependencies;

        iPythonWithPackages = pkgs.kernels.iPythonWith {
          name = "ms-thesis--env";
          python3 = python;
          packages = p: 
            let
              # Building the local package using the standard way.
              myPythonPackage = p.buildPythonPackage {
                pname = "my-python-package";
                version = "0.1.0";
                src = ./my-python-package;
              };
              # Getting dependencies using Poetry.
              poetryDeps =
                builtins.map (name: builtins.getAttr name p) depNames; # p : gets packages from 'python3 = python' ? maybe?
            in
              [ myPythonPackage ] ++ poetryDeps ;
        };
        jupyterEnvironment = pkgs.jupyterlabWith {
          kernels = [ iPythonWithPackages ];
          extraPackages = ps: [ps.hello];
        };
      in rec {
        apps.jupyterlab = {
          type = "app";
          program = "${jupyterEnvironment}/bin/jupyter-lab";
        };
        defaultApp = apps.jupyterlab;
        # devShell = jupyterEnvironment.env;
        devShell = pkgs.mkShell rec {
          inherit pkgs;
          buildInputs = [
            pkgs
            jupyterEnvironment
            pkgs.poetry
            #iJulia.runtimePackages
          ];

          #JULIA_DEPOT_PATH = "./.julia_depot";

          #shellHook = ''
          #'';
        };
      }
    );
}
# Initialize by making my-python-package and 
# $ nix shell nixpkgs#poetry
# $ poetry init
# $ poetry add ~ ~ ~
# inside my-python-package
