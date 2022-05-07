{ pkgs_import ? import ../nixpkgs {} }:
let
  pkgs_new = import ../nixpkgs {
    overlays = [
     # overlay = nixpkgs.lib.composeManyExtensions [
        (self: super: rec { 
          blas = (super.blas.override {
            blasProvider = self.mkl;
          }).overrideAttrs (oldAttrs: {
            buildInputs = (oldAttrs.buildInputs or [ ]) 
                          ++ self.lib.optional self.stdenv.hostPlatform.isDarwin self.fixDarwinDylibNames;
          });
          lapack = (super.lapack.override {
            lapackProvider = self.mkl;
          }).overrideAttrs (oldAttrs: {
            buildInputs = (oldAttrs.buildInputs or [ ]) 
                          ++ self.lib.optional self.stdenv.hostPlatform.isDarwin self.fixDarwinDylibNames;
          });
          blas_new = (super.blas.override {
            blasProvider = self.mkl;
          }).overrideAttrs (oldAttrs: {
            buildInputs = (oldAttrs.buildInputs or [ ]) 
                          ++ self.lib.optional self.stdenv.hostPlatform.isDarwin self.fixDarwinDylibNames;
          });
          lapack_new = (super.lapack.override {
            lapackProvider = self.mkl;
          }).overrideAttrs (oldAttrs: {
            buildInputs = (oldAttrs.buildInputs or [ ]) 
                          ++ self.lib.optional self.stdenv.hostPlatform.isDarwin self.fixDarwinDylibNames;
          });
          poetry2nix = super.poetry2nix.overrideScope' (p2nixself: p2nixsuper: {
          # pyself & pysuper refers to python packages
            defaultPoetryOverrides = p2nixsuper.defaultPoetryOverrides.extend (pyself: pysuper: {
              #importlib-metadata = pysuper.importlib-metadata.overridePythonAttrs ( old: {
              #  format = "pyproject";
              #});
              pyparsing = pysuper.pyparsing.overridePythonAttrs ( old: {
                buildInputs = (old.buildInputs or [ ]) ++ [ pyself.flit-core ];
              });
              pillow = pysuper.pillow.overridePythonAttrs ( old: {
                buildInputs = (old.buildInputs or [ ]) ++ [ self.xorg.libxcb ];
              });
              numpy = pysuper.numpy.overridePythonAttrs ( old:
                let
                  blas = blas_new;
                  lapack = lapack_new;
                  blasImplementation = "mkl";#nixpkgs.lib.nameFromURL blas.name "-";
                  cfg = super.writeTextFile {
                    name = "site.cfg";
                    text = (
                      super.lib.generators.toINI
                        { }
                        {
                          ${blasImplementation} = {
                            include_dirs = "${blas}/include";
                            library_dirs = "${blas}/lib";
                          } // super.lib.optionalAttrs (blasImplementation == "mkl") {
                            mkl_libs = "mkl_rt";
                            lapack_libs = "";
                          };
                        }
                    );
                  };
                in
                {
                  version = "1.22.3";
                  nativeBuildInputs = (old.nativeBuildInputs or [ ])
                                      ++ [ self.gfortran ];
                  buildInputs = (old.buildInputs or [ ]) 
                                ++ [ blas_new lapack_new pyself.emoji self.cowsay ];
                  enableParallelBuilding = true;
                  preBuild = ''
                    ln -s ${cfg} site.cfg
                  '';
                  passthru = old.passthru // {
                    blas = blas;
                    inherit blasImplementation cfg;
                  };
                }
              );
              #numpy = pysuper.numpy.overridePythonAttrs (
              #  old:
              #  let
              #    blas = blas_new;
              #    #lapack = lapack_new;
              #    blasImplementation = nixpkgs.lib.nameFromURL blas.name "-";
              #  in
              #  {
              #    buildInputs = (old.buildInputs or [ ]) ++ [ blas ];
              #    passthru = old.passthru // {
              #      blas = blas;
              #      inherit blasImplementation;# cfg;
              #    };
              #  }
              #);
              # numpy = pysuper.numpy.override {
              #   blas = blas_new;
              #   lapack = lapack_new;
              # }; # NOT WORKING
            });
          });
        })
        #(final: prev: {
        #  # The application
        #  myapp = prev.poetry2nix.mkPoetryApplication {
        #    projectDir = ./.;
        #  };
        #})
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

  #python_test = pkgs_new.poetry2nix.mkPoetryEnv {
  #  projectDir = ./.;
  #};
  #jupyter = import (builtins.fetchGit {
  #  url = https://github.com/tweag/jupyterWith;
  #  # Example working revision, check out the latest one.
  #  #rev = "45f9a774e981d3a3fb6a1e1269e33b4624f9740e";
  #}) {};

  #      pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
  #      depNames = builtins.attrNames pyproject.tool.poetry.dependencies;

  #      iPythonWithPackages = jupyter.kernels.iPythonWith {
  #        name = "ms-thesis--env";
  #        python3 = python_test;
  #        packages = p: 
  #          let
  #            # Building the local package using the standard way.
  #            myPythonPackage = p.buildPythonPackage {
  #              pname = "my-python-package";
  #              version = "0.2.0";
  #              src = ./my-python-package;
  #            };
  #            # Getting dependencies using Poetry.
  #            poetryDeps =
  #              builtins.map (name: builtins.getAttr name p) depNames; 
  #              # p : gets packages from 'python3 = python' ? maybe?
  #          in
  #            # [ p.emoji ] ++ # adds nixpkgs.url version  python pkgs.
  #            [ myPythonPackage ] ++ poetryDeps; ### ++ (poetryExtraDeps p);
  #      };
  #      jupyterEnvironment = jupyter.jupyterlabWith {
  #        kernels = [ iPythonWithPackages ];
  #        extraPackages = ps: [ps.hello ];
  #      };
in python-with-my-packages.env #python_test.env #jupyterEnvironment.env #myAppEnv.env
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
