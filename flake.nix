{
  description = "MS thesis environment";
  inputs = {
    nixpkgs.url = "github:sepiabrown/nixpkgs/test_mkl_on_server_echo1";#NixOS/nixpkgs"; # poetry doesn't work at nixos-20.09
    #nixpkgs_1703.url = "nixpkgs/1849e695b00a54cda86cb75202240d949c10c7ce"; # poetry doesn't work at nixos-20.09
    jupyterWith = {
      url = "github:tweag/jupyterWith"; 
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ self, nixpkgs, jupyterWith, flake-utils }:
    {
      overlay = nixpkgs.lib.composeManyExtensions ([
        (self: super: { 
          blas_custom = (super.blas.override {
            blasProvider = self.mkl;
          }).overrideAttrs (oldAttrs: {
            buildInputs = (oldAttrs.buildInputs or [ ]) 
                          ++ self.lib.optional self.stdenv.hostPlatform.isDarwin self.fixDarwinDylibNames;
          });
          lapack_custom = (super.lapack.override {
            lapackProvider = self.mkl;
          }).overrideAttrs (oldAttrs: {
            buildInputs = (oldAttrs.buildInputs or [ ]) 
                          ++ self.lib.optional self.stdenv.hostPlatform.isDarwin self.fixDarwinDylibNames;
          });
          python3 = super.python3.override (old: { # for jupyterWith!
            packageOverrides = 
              super.lib.composeExtensions
                (old.packageOverrides or (_: _: {}))
                (python-self: python-super: {
                  httpx = python-super.httpx.overridePythonAttrs (old: { # for jupyterlab -> .. -> falcon
                    doCheck = false;
                  });
                  httplib2 = python-super.httplib2.overridePythonAttrs ( old: {
                    doCheck = false;
                  });
                  numpy = python-super.numpy.overridePythonAttrs ( old:
                    let
                      blas = self.blas_custom; # not super.blas
                      lapack = self.lapack_custom; # not super.lapack
                      blasImplementation = nixpkgs.lib.nameFromURL blas.name "-";
                      cfg = super.writeTextFile {
                        name = "site.cfg";
                        text = (
                          nixpkgs.lib.generators.toINI
                            { }
                            {
                              ${blasImplementation} = {
                                include_dirs = "${blas}/include";
                                library_dirs = "${blas}/lib";
                              } // nixpkgs.lib.optionalAttrs (blasImplementation == "mkl") {
                                mkl_libs = "mkl_rt";
                                lapack_libs = "";
                              };
                            }
                        );
                      };
                    in
                    {
                      nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ self.gfortran ];
                      buildInputs = (old.buildInputs or [ ]) ++ [ blas lapack ];
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
                  #astroid = python-super.astroid.overridePythonAttrs ( old: {
                  #  version = "2.11.2";
                  #  buildInputs = (old.buildInputs or [ ]) ++ [ python-self.wrapt ];
                  #  propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ python-self.wrapt ];
                  #  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ python-self.wrapt ];
                  #  propagatedNativeBuildInputs = (old.propagatedNativeBuildInputs or [ ]) ++ [ python-self.wrapt ];
                  #});
                });
          });
          poetry2nix = super.poetry2nix.overrideScope' (p2nixself: p2nixsuper: {
          # pyself & pysuper refers to python packages
            defaultPoetryOverrides = p2nixsuper.defaultPoetryOverrides.extend (pyself: pysuper: {
              #importlib-metadata = pysuper.importlib-metadata.overridePythonAttrs ( old: {
              #  format = "pyproject";
              #});
              astroid = pysuper.astroid.overridePythonAttrs ( old: rec {
                version = "2.11.2";
                  src = self.fetchFromGitHub {
                    owner = "PyCQA";
                    repo = "astroid";
                    rev = "v${version}";
                    sha256 = "sha256-adnvJCchsMWQxsIlenndUb6Mw1MgCNAanZcTmssmsEc=";
                  };
                #buildInputs = (old.buildInputs or [ ]) ++ [ pyself.wrapt ];
                #propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ pyself.wrapt ];
                #nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pyself.wrapt ];
                #propagatedNativeBuildInputs = (old.propagatedNativeBuildInputs or [ ]) ++ [ pyself.wrapt ];
              });
              tensorflow = pysuper.tensorflow.overridePythonAttrs ( old: {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pyself.wheel ];
              });
              scipy = pysuper.scipy.overridePythonAttrs ( old: {
                doCheck = false;
              });
              tensorflow-io-gcs-filesystem = pysuper.tensorflow-io-gcs-filesystem.overridePythonAttrs ( old: {
                buildInputs = (old.buildInputs or [ ]) ++ [ self.libtensorflow ];
              });
              libclang = pysuper.libclang.overridePythonAttrs ( old: {
                buildInputs = (old.buildInputs or [ ]) ++ [ self.zlib ];
              });
              pyparsing = pysuper.pyparsing.overridePythonAttrs ( old: {
                buildInputs = (old.buildInputs or [ ]) ++ [ pyself.flit-core ];
              });
              pillow = pysuper.pillow.overridePythonAttrs ( old: {
                buildInputs = (old.buildInputs or [ ]) ++ [ self.xorg.libxcb ];
              });
              numpy = pysuper.numpy.overridePythonAttrs ( old:
                let
                  blas = self.blas_custom; # not super.blas
                  lapack = self.lapack_custom; # not super.lapack
                  blasImplementation = nixpkgs.lib.nameFromURL blas.name "-";
                  cfg = super.writeTextFile {
                    name = "site.cfg";
                    text = (
                      nixpkgs.lib.generators.toINI
                        { }
                        {
                          ${blasImplementation} = {
                            include_dirs = "${blas}/include";
                            library_dirs = "${blas}/lib";
                          } // nixpkgs.lib.optionalAttrs (blasImplementation == "mkl") {
                            mkl_libs = "mkl_rt";
                            lapack_libs = "";
                          };
                        }
                    );
                  };
                in
                {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ self.gfortran ];
                  buildInputs = (old.buildInputs or [ ]) ++ [ blas lapack ];
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
            });
          });
        })
        #(final: prev: {
        #  # The application
        #  myapp = prev.poetry2nix.mkPoetryApplication {
        #    projectDir = ./.;
        #  };
        #})
        #jupyterWith.overlays.jupyterWith
        #jupyterWith.overlays.haskell
        #jupyterWith.overlays.python
      ] ++ (builtins.attrValues jupyterWith.overlays));
    } // (flake-utils.lib.eachSystem [ "x86_64-linux" "x86_64-darwin" ] (system:
      let
        pkgs = import nixpkgs {
          system = system;
          config.allowUnfree = true;
          overlays = [ self.overlay ];
          #overlays = (builtins.attrValues jupyterWith.overlays) ++ [ self.overlay ]; # [ (import ./haskell-overlay.nix) ];
        };

        python_custom = pkgs.poetry2nix.mkPoetryEnv rec {
          projectDir = ./.;
          pyproject = projectDir + "/pyproject.toml";
          poetrylock = projectDir + "/poetry.lock";

          #editablePackageSources = {
          #  ronald_bdl = "${builtins.getEnv "HOME"}/MS-Thesis/my-python-package/ronald_bdl";
          #ronald_bdl = ./my-python-package/ronald_bdl;
          #};
          ### extraPackages = poetryExtraDeps;

          # For jupyter-lab's use, there is no need to add packages that 
          # are frequently edited.
          # Just import by using
          #
          # import sys
          # sys.path.append("/home/sepiabrown/MS-Thesis/my-python-package/")
          # import ronald_bdl
        };

        #pkgs_1703 = import (builtins.fetchGit {
        #  # Descriptive name to make the store path easier to identify
        #  name = "nixos-1703";
        #  url = "https://github.com/nixos/nixpkgs/";
        #  # Commit hash for nixos-unstable as of 2018-09-12
        #  # `git ls-remote https://github.com/nixos/nixpkgs nixos-unstable`
        #  #allRefs = true;
        #  ref = "release-17.09";
        #  rev = "3ba3d8d8cbec36605095d3a30ff6b82902af289c";
        #  #rev = "1849e695b00a54cda86cb75202240d949c10c7ce"; 1703
        #  #rev = "a7ecde854aee5c4c7cd6177f54a99d2c1ff28a31"; 2111
        #}) { system = system; };

        #python-with-my-packages = (pkgs_1703.python27.withPackages (p: with p; [
        #  cython
        #  numpy
        #  scipy
        #  scikitimage
        #  matplotlib
        #  ipython
        #  h5py
        #  #leveldb
        #  networkx
        #  nose
        #  pandas
        #  #python-dateutil
        #  pyparsing
        #  dateutil
        #  protobuf
        #  gflags
        #  pyyaml
        #  pillow
        #  six
        #])).override (args: { ignoreCollisions = true; });

        #pkgs_1703.poetry2nix.mkPoetryEnv rec {
        #  projectDir = ./.;
        #  python = pkgs.python31;
        #};

        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
        depNames = builtins.attrNames pyproject.tool.poetry.dependencies;

        iPythonWithPackages = pkgs.kernels.iPythonWith {
          name = "ms-thesis--env";
          python3 = python_custom;
          packages = p: 
            let
              # Building the local package using the standard way.
              myPythonPackage = p.buildPythonPackage {
                pname = "my-python-package";
                version = "0.2.0";
                src = ./my-python-package;
              };
              # Getting dependencies using Poetry.
              poetryDeps =
                builtins.map (name: builtins.getAttr name p) depNames; 
                # p : gets packages from 'python3 = python' ? maybe?
            in
              # [ p.emoji ] ++ # adds nixpkgs.url version  python pkgs.
              [ myPythonPackage ] ++ poetryDeps; ### ++ (poetryExtraDeps p);
        };
        jupyterEnvironment = pkgs.jupyterlabWith {
          kernels = [ iPythonWithPackages ];
          extraPackages = ps: [ ];
        };
      in rec {
        apps.jupyterlab = {
          type = "app";
          program = "${jupyterEnvironment}/bin/jupyter-lab";
        };
        defaultApp = apps.jupyterlab;
        inherit nixpkgs python_custom;
        #devShell = python_test.env.overrideAttrs (old: {
        #  nativeBuildInputs = with pkgs; old.nativeBuildInputs ++ [
        #    jupyterEnvironment
        #    poetry
        #    (lib.getBin caffe)
        #  ];
        #});
        devShell = pkgs.mkShell rec {
          buildInputs = [
            # stdenv reference :
            # https://discourse.nixos.org/t/nixos-with-poetry-installed-pandas-libstdc-so-6-cannot-open-shared-object-file/8442/3
            # https://nixos.wiki/wiki/Packaging/Quirks_and_Caveats#ImportError:_libstdc.2B.2B.so.6:_cannot_open_shared_object_file:_No_such_file
            pkgs.stdenv.cc.cc.lib
          ];
          nativeBuildInputs = [
            jupyterEnvironment
            pkgs.poetry
            (pkgs.lib.getBin pkgs.caffe)
            #python-with-my-packages
            #poetry_31
            #python_custom
            #(pkgs.lib.getBin python_custom)

            #iJulia.runtimePackages
          ];
          #JULIA_DEPOT_PATH = "./.julia_depot";

          #export PYTHONPATH="${python-with-my-packages}/${python-with-my-packages.sitePackages}:$PYTHONPATH"
          #shellHook = ''
          #  export PYTHONPATH="${python-with-my-packages}/lib/python2.7/site-packages:$PYTHONPATH"
          #'';
        };
      }
    ));
}
# Initialize by 
# $ nix shell nixpkgs#poetry
# $ poetry init
# $ poetry add ~ ~ ~
# inside MS-Thesis
# On bayes-lab server, mkl is already installed.
# Nix captures the existance of mkl, but without explicit declaration of mkl in flake.nix,
# build process fails with error "cannot find -lmkl_rt"
# Thus, through overlay, we need to override blas and lapack which use mkl.
# In overlay, names should be blas, lapack, not blas_new lapack_new for them to be overriden globally.
#
# ssh -f -p 7777 sepiabrown@snubayes.duckdns.org "./.cargo/bin/nix-user-chroot ~/.nix bash -l -c 'nix run ./MS-Thesis -- --port 3333'"
# ssh -N -p7777 -L3333:localhost:3333 sepiabrown@snubayes.duckdns.org
# 
# nohup ssh -f -p 7777 -L 3333:localhost:3333 sepiabrown@snubayes.duckdns.org "./.cargo/bin/nix-user-chroot ~/.nix bash -l -c 'nix run ./MS-Thesis -- --port 3333'" > /dev/null
