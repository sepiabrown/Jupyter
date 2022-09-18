{
  description = "MS thesis environment";
  inputs = {
    #nixpkgs_edge.url = "github:sepiabrown/nixpkgs/poetry_edge_test2";#test_mkl_on_server_echo1";#NixOS/nixpkgs"; # poetry doesn't work at nixos-20.09
    #nixpkgs.url = "github:sepiabrown/nixpkgs/download_fix_220721"; #test_mkl_on_server_echo1";#NixOS/nixpkgs"; # poetry doesn't work at nixos-20.09
    nixpkgs.url = "nixpkgs/nixos-22.05"; 
    #nixpkgs_2111.url = "nixpkgs/nixos-21.11"; 
    #nixpkgs_2009.url = "nixpkgs/nixos-20.09"; # poetry doesn't work at nixos-20.09
    #nixpkgs_1703.url = "nixpkgs/1849e695b00a54cda86cb75202240d949c10c7ce"; # poetry doesn't work at nixos-20.09
    jupyterWith = {
      url = "github:sepiabrown/jupyterWith/python39_and_poetry2nix"; #/environment_variables_test"; 
      #inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
      #inputs.nixpkgs.follows = "nixpkgs";
    };
    poetry2nix_nix_community = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    dpaetzel.url = "github:dpaetzel/overlays/master";
    nixgl = {
      url = "github:sepiabrown/nixGL";
    };
  };

  outputs = inputs: with inputs; # inputs@{ self, nixpkgs, jupyterWith, flake-utils, dpaetzel, ... }:
    let
      LD_LIBRARY_PATH = ''
        ${nixpkgs.lib.makeLibraryPath [ self.pkgs.x86_64-linux.cudaPackages_11_6.cudatoolkit "${self.pkgs.x86_64-linux.cudaPackages_11_6.cudatoolkit}" self.pkgs.x86_64-linux.cudaPackages_11_6.cudnn self.pkgs.x86_64-linux.nvidia_custom ]}:$LD_LIBRARY_PATH
      '';
    in
    {
      overlay = nixpkgs.lib.composeManyExtensions ([
        #poetry2nix_nix_community.overlay
        (final: prev: {
          blas_custom = (prev.blas.override {
            blasProvider = final.mkl;
          }).overrideAttrs (oldAttrs: {
            buildInputs = (oldAttrs.buildInputs or [ ])
            ++ final.lib.optional final.stdenv.hostPlatform.isDarwin final.fixDarwinDylibNames;
          });
          lapack_custom = (prev.lapack.override {
            lapackProvider = final.mkl;
          }).overrideAttrs (oldAttrs: {
            buildInputs = (oldAttrs.buildInputs or [ ])
            ++ final.lib.optional final.stdenv.hostPlatform.isDarwin final.fixDarwinDylibNames;
          });
          # For jep - openjdk8 - liberation_ttf - ... - numpy -> mkl error
          liberation_ttf = prev.liberation_ttf.override {
            #python3 = self.python_custom.x86_64-linux;
            python3 = final.python39;
          };
          libtensorflow = self.python_custom.x86_64-linux.pkgs.tensorflow.passthru.libtensorflow;
          nvidia_custom = prev.linuxPackages.nvidia_x11.overrideAttrs (oldAttrs: rec {
            version = "495.29.05";
            src = builtins.fetchurl {
              url = "https://us.download.nvidia.com/XFree86/Linux-x86_64/${version}/NVIDIA-Linux-x86_64-${version}.run";
              sha256 = "sha256-9yVLl9QAxpJQR5ZJb059j2TpOx4xxCeGCk8hmhhvEl4="; #prev.lib.fakeSha256;
            };
          });
          poetry2nix = prev.poetry2nix.overrideScope' (p2nixfinal: p2nixprev: {
            # pyfinal & pyprev refers to python packages
            defaultPoetryOverrides = (p2nixprev.defaultPoetryOverrides.extend (pyfinal: pyprev:
              #let
              #  torch_custom = nixpkgs.lib.makeOverridable
              #    (
              #      { enableCuda ? true
              #      , cudaPackages ? final.cudaPackages #, cudatoolkit ? pkgs.cudatoolkit_10_1
              #      , cudaArchList ? null
              #      , pkg ? pyprev.torch
              #      }:
              #      let
              #        inherit (cudaPackages) cudatoolkit cudnn nccl cuda_nvcc;

              #        cudatoolkit_joined = final.symlinkJoin {
              #          name = "${cudatoolkit.name}-unsplit";
              #          # nccl is here purely for semantic grouping it could be moved to nativeBuildInputs
              #          paths = [ cudatoolkit.out cudatoolkit.lib nccl.dev nccl.out ];
              #        };

              #        cudaCapabilities = rec {
              #          cuda9 = [
              #            "3.5"
              #            "5.0"
              #            "5.2"
              #            "6.0"
              #            "6.1"
              #            "7.0"
              #            "7.0+PTX" # I am getting a "undefined architecture compute_75" on cuda 9
              #            # which leads me to believe this is the final cuda-9-compatible architecture.
              #          ];

              #          cuda10 = cuda9 ++ [
              #            "7.5"
              #            "7.5+PTX" # < most recent architecture as of cudatoolkit_10_0 and pytorch-1.2.0
              #          ];

              #          cuda11 = cuda10 ++ [
              #            "8.0"
              #            "8.0+PTX" # < CUDA toolkit 11.0
              #            "8.6"
              #            "8.6+PTX" # < CUDA toolkit 11.1
              #          ];
              #        };
              #        final_cudaArchList =
              #          if !enableCuda || cudaArchList != null
              #          then cudaArchList
              #          else cudaCapabilities."cuda${nixpkgs.lib.versions.major cudatoolkit.version}";
              #      in
              #      pkg.overrideAttrs (old: {
              #        src_test = old.src;
              #        preferWheels = true;
              #        dontStrip = false;
              #        format = "wheels";

              #        preConfigure = nixpkgs.lib.optionalString (!enableCuda) ''
              #          export USE_CUDA=0
              #        '' + nixpkgs.lib.optionalString enableCuda ''
              #          export TORCH_CUDA_ARCH_LIST="${nixpkgs.lib.strings.concatStringsSep ";" final_cudaArchList}"
              #          export CC=${cudatoolkit.cc}/bin/gcc CXX=${cudatoolkit.cc}/bin/g++
              #        '' + ''export LD_LIBRARY_PATH='' + LD_LIBRARY_PATH +
              #        nixpkgs.lib.optionalString (enableCuda && cudnn != null) ''
              #          export CUDNN_INCLUDE_DIR=${cudnn}/include
              #        ''; # enableCuda ${cudatoolkit}/targets/x86_64-linux/lib

              #        # patchelf --set-rpath "${lib.makeLibraryPath [ stdenv.cc.cc.lib ]}"
              #        preFixup = nixpkgs.lib.optionalString (!enableCuda) ''
              #          # For some reason pytorch retains a reference to libcuda even if it
              #          # is explicitly disabled with USE_CUDA=0.
              #          find $out -name "*.so" -exec ${nixpkgs.patchelf}/bin/patchelf --remove-needed libcuda.so.1 {} \;
              #        '';
              #        nativeBuildInputs =
              #          (old.nativeBuildInputs or [ ])
              #          ++ [ final.autoPatchelfHook ]
              #          ++ nixpkgs.lib.optionals enableCuda [ cudatoolkit_joined final.addOpenGLRunpath ];
              #        buildInputs =
              #          (old.buildInputs or [ ])
              #          ++ [ pyfinal.typing-extensions pyfinal.pyyaml ]
              #          ++ nixpkgs.lib.optionals enableCuda [
              #            final.nvidia_custom
              #            nccl.dev
              #            nccl.out
              #            cudatoolkit
              #            cudnn
              #            cuda_nvcc
              #            final.magma
              #            nccl
              #          ];
              #        propagatedBuildInputs = [
              #          pyfinal.numpy
              #          pyfinal.future
              #          pyfinal.typing-extensions
              #        ];
              #      })
              #    )
              #    { };
              #in
              {
                #importlib-metadata = pyprev.importlib-metadata.overridePythonAttrs ( old: {
                #  format = "pyproject";
                #});
                astroid = pyprev.astroid.overridePythonAttrs (old: rec {
                  version = "2.11.2";
                  src = final.fetchFromGitHub {
                    owner = "PyCQA";
                    repo = "astroid";
                    rev = "v${version}";
                    sha256 = "sha256-adnvJCchsMWQxsIlenndUb6Mw1MgCNAanZcTmssmsEc=";
                  };
                  #buildInputs = (old.buildInputs or [ ]) ++ [ pyfinal.wrapt ];
                  #propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ pyfinal.wrapt ];
                  #nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pyfinal.wrapt ];
                  #propagatedNativeBuildInputs = (old.propagatedNativeBuildInputs or [ ]) ++ [ pyfinal.wrapt ];
                });
                # gast 0.5.0 needed by beniget needed by pythran needed by scipy
                # but tensorflow needs gast <= 0.4.0
                beniget = pyprev.beniget.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ pyprev.gast ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.gast_5 ]);
                });

                cython = pyprev.cython.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = [ pyfinal.setuptools ];
                });
                Cython = pyfinal.cython;

                gast_5 = pyprev.gast.overridePythonAttrs (old: rec {
                  pname = "gast";
                  version = "0.5.0";

                  src = final.fetchFromGitHub {
                    owner = "serge-sans-paille";
                    repo = pname;
                    rev = version;
                    sha256 = "sha256-fGkr1TIn1NPlRh4hTs9Rh/d+teGklrz+vWKCbacZT2M=";
                  };
                });
                httpx = pyprev.httpx.overridePythonAttrs (old: {
                  # for jupyterlab -> .. -> falcon
                  doCheck = false;
                });
                jax = (pyfinal.python_selected.pkgs.jax.override {
                  inherit (pyfinal)
                    absl-py
                    numpy
                    opt-einsum
                    scipy
                    typing-extensions
                    jaxlib
                    #pytest-xdist # duplicated packages error
                    ;
                  blas = final.blas_custom;
                  lapack = final.lapack_custom;
                }).overridePythonAttrs (old: rec {
                  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [ pyfinal.jaxlib ]);
                  doCheck = false;
                });
                jaxlib = pyfinal.python_selected.pkgs.jaxlib.override {
                  #inherit (final)
                  #  # Build-time dependencies:
                  #  addOpenGLRunpath
                  #  bazel_5
                  #  binutils
                  #  buildBazelPackage
                  #  fetchFromGitHub
                  #  git
                  #  jsoncpp
                  #  symlinkJoin
                  #  which
                  #  ;
                  inherit (pyfinal)
                    # Build-time dependencies:
                    cython
                    pybind11
                    setuptools
                    wheel

                    # Python dependencies:
                    absl-py
                    flatbuffers
                    numpy
                    scipy
                    six

                    # Runtime dependencies:
                    #double-conversion
                    #giflib
                    #grpc
                    #libjpeg_turbo
                    python
                    #snappy
                    #zlib
                    ;
                  cudaSupport = true;
                  cudaPackages = final.cudaPackages_11_6;
                  mklSupport = true;
                };
                #.overridePythonAttrs ( old: {
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x (with final.python3.pkgs; [ scipy numpy six ])) ((old.propagatedBuildInputs or [ ]) ++ [ ]);
                #});
                jep = pyprev.jep.overridePythonAttrs (old: {
                  propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [
                    pyfinal.setuptools
                    pyfinal.numpy
                  ];
                  buildInputs = (old.buildInputs or [ ]) ++ [
                    final.glibc
                    final.openjdk8
                  ];
                });
                jupyterlab = pyprev.jupyterlab.overridePythonAttrs (oldAttrs: {
                  makeWrapperArgs = (oldAttrs.makeWrapperArgs or [ ]) ++ [
                    "--set LD_LIBRARY_PATH"
                    LD_LIBRARY_PATH
                    "--set TF_ENABLE_ONEDNN_OPTS 0" # when using GPU, oneDNN off is recommended 
                    "--set XLA_FLAGS --xla_gpu_cuda_data_dir=${final.cudaPackages_11_6.cudatoolkit}"
                    #"--set AESARA_FLAGS device=cuda0,dnn__base_path=${final.cudaPackages_11_6.cudnn},blas__ldflags=-lblas,dnn__library_path=${final.cudaPackages_11_6.cudnn}/lib,dnn__include_path=${final.cudaPackages_11_6.cudnn}/include"#${nixpkgs.lib.makeLibraryPath [ final.cudaPackages.cudnn ]}" #,cuda__root=${final.cudaPackages_11_6.cudatoolkit}
                    #"--set CUDA_HOME ${final.cudaPackages_11_6.cudatoolkit}"
                    #"--set CUDA_INC_DIR ${final.cudaPackages_11_6.cudatoolkit}/include"
          #export CUDA_PATH=${pkgs.cudatoolkit}
          #export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
                  ];
                });
                jsonschema = pyprev.jsonschema.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [
                    pyfinal.hatchling
                    # 'hatchling.build' has no attribute 'prepare_metadata_for_build_wheel': needs hatch-vcs not importlib-metadata
                    pyfinal.hatch-vcs
                    pyfinal.attrs
                    pyfinal.pyrsistent
                  ]);
                });
                #kaggle = pyprev.kaggle.overridePythonAttrs (old: {
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ pyprev.tqdm ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.tqdm pyfinal.tqdm_custom ]);
                ###  #propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ pyfinal.t];
                #});
                #kaggle = (final.python3.pkgs.kaggle.override {
                #  inherit (pyfinal) 
                #    python-dateutil
                #    python-slugify
                #    six
                #    requests
                #    #tqdm_custom 
                #    urllib3;
                #  tqdm = pyfinal.tqdm_custom; # tqdm needs overriden numpy
                #});
                #libgpuarray = pyprev.libgpuarray.overridePythonAttrs ( old: {
                #  libraryPath = nixpkgs.lib.makeLibraryPath (with final.cudaPackages_11_6; [ cudnn cudatoolkit.lib cudatoolkit.out final.clblas final.ocl-icd ]);
                #  buildInputs = (old.buildInputs or [ ]) ++ [ final.cudaPackages_11_6.cudatoolkit final.cudaPackages_11_6.cudnn pyfinal.nose];
                #  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                #    final.cudaPackages_11_6.cudatoolkit final.cudaPackages_11_6.cudnn pyfinal.nose
                #  ];
                #});
                mkl-service = pyprev.mkl-service.overridePythonAttrs (old: {
                  patchPhase = ''
                    substituteInPlace mkl/_mkl_service.pyx \
                      --replace "'avx512_e4': mkl.MKL_ENABLE_AVX512_E4," " " \
                      --replace "'avx2_e1': mkl.MKL_ENABLE_AVX2_E1," " "

                    substituteInPlace mkl/_mkl_service.pxd \
                      --replace "int MKL_ENABLE_AVX512_E4" " " \
                      --replace "int MKL_ENABLE_AVX2_E1" " "
                  '';
                });
                # numba needs setuptools<60 but jupyter-packaging needs setuptools>=60
                #numba = pyprev.numba.override {
                #  setuptools = nixpkgs_2111.legacyPackages.x86_64-linux.python39.pkgs.setuptools;
                #};


                numpy = pyprev.numpy.overridePythonAttrs (old:
                  let
                    blas = final.mkl; # not prev.blas
                    blasImplementation = nixpkgs.lib.nameFromURL blas.name "-";
                    cfg = prev.writeTextFile {
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
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.gfortran ];
                    buildInputs = (old.buildInputs or [ ]) ++ [ blas ];
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
                pillow = pyprev.pillow.overridePythonAttrs (old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ final.xorg.libxcb ];
                });
                #poetry = pyprev.poetry.overridePythonAttrs (old: rec {
                #  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ pyfinal.setuptools ]) ((old.nativeBuildInputs or [ pyprev.setuptools ]) ++ [
                #  ]);
                #});
                poetry-core = pyprev.poetry-core.overridePythonAttrs (old: rec {
                  # poetry 1.2.0b1
                  version = "1.1.0b3"; # poetry-core version is different from poetry
                  src = final.fetchFromGitHub {
                    owner = "python-poetry";
                    repo = "poetry-core";
                    rev = version;
                    sha256 = "sha256-clQw8twOUYL8Ew/FioKwOIJwIhsVPuyF5McVR2zzrO4=";
                  };
                  doCheck = false;
                  postPatch = nixpkgs.lib.optionalString (nixpkgs.lib.versionOlder final.python.version "3.8") ''
                    # remove >1.0.3
                    substituteInPlace pyproject.toml \
                      --replace 'importlib-metadata = {version = "^1.7.0", python = "~2.7 || >=3.5, <3.8"}' \
                        'importlib-metadata = {version = ">=1.7.0", python = "~2.7 || >=3.5, <3.8"}'
                  '';

                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                    #pyfinal.intreehooks
                  ];

                  propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ nixpkgs.lib.optionals (nixpkgs.lib.versionOlder final.python.version "3.8") [
                    #`pyfinal.importlib-metadata
                  ] ++ nixpkgs.lib.optionals pyfinal.isPy27 [
                    pyfinal.pathlib2
                    pyfinal.typing
                  ];

                  checkInputs = [
                    nixpkgs.git
                    pyfinal.pep517
                    pyfinal.pytest-mock
                    pyfinal.pytestCheckHook
                    pyfinal.tomlkit
                    pyfinal.virtualenv
                  ];
                  # 1.1.13
                  # "Vendor" dependencies (for build-system support)
                  #postPatch = ''
                  #  echo "import sys" >> poetry/__init__.py
                  #  for path in $propagatedBuildInputs; do
                  #      echo "sys.path.insert(0, \"$path\")" >> poetry/__init__.py
                  #  done
                  #'';

                  ## Propagating dependencies leads to issues downstream
                  ## We've already patched poetry to prefer "vendored" dependencies
                  #postFixup = ''
                  #  rm $out/nix-support/propagated-build-inputs
                  #'';
                });
                poetry-plugin-export = pyprev.buildPythonPackage rec {
                  pname = "poetry-plugin-export";
                  version = "1.0.5";
                  src = pyfinal.fetchPypi {
                    inherit pname version;
                    hash = "sha256-53likuqvrBMWFJ86gHCSPCoiFMmNBG3hJGtNjrCwyEs=";
                  };
                  nativeBuildInputs = [ pyfinal.pythonRelaxDepsHook ];
                  pythonRemoveDeps = [ "poetry" ];
                  buildInputs = [
                    pyfinal.poetry-core
                  ];
                  doCheck = false;
                };

                ## pymc4
                pymc = pyprev.pymc.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [
                    pyfinal.mkl-service
                    #    pyfinal.aeppl
                  ]);
                });
                pymc-nightly =
                  let
                    blas = final.blas_custom; # not prev.blas
                    lapack = final.lapack_custom; # not prev.lapack
                    blasImplementation = nixpkgs.lib.nameFromURL blas.name "-";
                    cfg = prev.writeTextFile {
                      name = "site.cfg";
                      text = (
                        nixpkgs.lib.generators.toINI
                          { }
                          {
                            ${blasImplementation} = {
                              include_dirs = "${blas}/include";
                              library_dirs = "${blas}/lib";
                            } // nixpkgs.lib.optionalAttrs (blas.implementation == "mkl") {
                              mkl_libs = "mkl_rt";
                              lapack_libs = "";
                            };
                          }
                      );
                    };
                  in
                  pyprev.buildPythonPackage rec {
                    pname = "pymc-nightly";
                    version = "4.0.0b2.dev20220304";

                    nativeBuildInputs = [ final.gfortran ];
                    buildInputs = [ blas lapack ];
                    enableParallelBuilding = true;
                    preBuild = ''
                      ln -s ${cfg} site.cfg
                    '';

                    meta.broken = false;

                    src = pyprev.fetchPypi {
                      inherit pname version;
                      sha256 = "sha256-U5s/HL8h7hLAOXWFlyvmbToqiZfEfRes3i57L+eCSJs=";
                    };

                    propagatedBuildInputs = [
                      pyfinal.arviz
                      pyfinal.cachetools
                      pyfinal.fastprogress
                      pyfinal.h5py
                      pyfinal.joblib
                      pyfinal.packaging
                      pyfinal.pandas
                      pyfinal.patsy
                      pyfinal.semver
                      pyfinal.tqdm
                      pyfinal.typing-extensions
                      pyfinal.jaxlib

                      pyfinal.cloudpickle

                      pyfinal.aeppl
                      pyfinal.aesara
                      pyfinal.numpy
                      blas
                      lapack
                      pyfinal.mkl-service
                    ];

                    # From the pymc3 Nix package:
                    # “The test suite is computationally intensive and test failures are not
                    # indicative for package usability hence tests are disabled by default.”
                    doCheck = false;
                    pythonImportsCheck = [ "pymc" ];

                    # From the pymc3 Nix package:
                    # “For some reason tests are run as a part of the *install* phase if
                    # enabled.  Theano writes compiled code to ~/.theano hence we set
                    # $HOME.”
                    preInstall = "export HOME=$(mktemp -d)";
                    postInstall = "rm -rf $HOME";

                    checkInputs = with pyprev; [ pytest pytest-cov ];
                  };

                python_selected = prev.python39;

                # pythran needed by scipy
                pythran = pyprev.pythran.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ pyprev.gast ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.gast_5 ]);
                  # needed for scipy 0.18.1
                  #pname = "pythran";
                  #version = "0.10.0";

                  #src = final.fetchFromGitHub {
                  #  owner = "serge-sans-paille";
                  #  repo = "pythran";
                  #  rev = version;
                  #  sha256 = "sha256-BLgMcK07iuRPBJqQpLXGnf79KgRsTqygCSeusLqkfxc=";
                  #};
                });

                pyzmq = pyprev.pyzmq.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [
                    pyfinal.packaging
                    #    pyfinal.aeppl
                  ]);
                });

                scipy = pyprev.scipy.overridePythonAttrs (old: {
                  doCheck = false;
                });

                tensorflow-gpu = pyprev.tensorflow.override {
                  cudaSupport = true;
                  mklSupport = true;
                  mkl = final.mkl;
                };
                #= pyprev.tensorflow-gpu.overridePythonAttrs (old: { # tensorflow-gpu doesn't exist! Always search at hound!
                #  #buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [ pyfinal.tensorboard ]);
                #  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.nativeBuildInputs or [ ]) ++ [ pyfinal.tensorboard ]);
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ pyprev.tensorboard ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.wheel ]); # pyprev.gast pyfinal.gast_4 
                #});

                tensorflow-io-gcs-filesystem = pyprev.tensorflow-io-gcs-filesystem.overridePythonAttrs (old: {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.numpy ]);
                  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [ final.libtensorflow ]);
                });

                # needed for pymc3
                theano-pymc = pyprev.theano-pymc.overridePythonAttrs (old:
                  let
                    wrapped = command: buildTop: buildInputs:
                      final.runCommandCC "${command}-wrapped" { inherit buildInputs; } ''
                        type -P '${command}' || { echo '${command}: not found'; exit 1; }
                        cat > "$out" <<EOF
                        #!$(type -P bash)
                        $(declare -xp | sed -e '/^[^=]\+="\('"''${NIX_STORE//\//\\/}"'\|[^\/]\)/!d')
                        declare -x NIX_BUILD_TOP="${buildTop}"
                        $(type -P '${command}') "\$@"
                        EOF
                        chmod +x "$out"
                      '';

                    # Theano spews warnings and disabled flags if the compiler isn't named g++
                    cxx_compiler_name =
                      if final.stdenv.cc.isGNU then "g++" else
                      if final.stdenv.cc.isClang then "clang++" else
                      throw "Unknown C++ compiler";
                    cxx_compiler = wrapped cxx_compiler_name "\\$HOME/.theano"
                      ([ pyfinal.libgpuarray final.cudaPackages_11_6.cudnn final.cudaPackages_11_6.cudatoolkit ]);

                  in
                  {
                    postPatch = ''
                      substituteInPlace theano/configdefaults.py \
                        --replace 'StrParam(param, is_valid=warn_cxx)' 'StrParam('\'''${cxx_compiler}'\''', is_valid=warn_cxx)' \
                        --replace 'rc == 0 and config.cxx != ""' 'config.cxx != ""'
                      substituteInPlace theano/configdefaults.py \
                        --replace 'StrParam(get_cuda_root)' 'StrParam('\'''${final.cudaPackages_11_6.cudatoolkit}'\''')'
                      substituteInPlace theano/configdefaults.py \
                        --replace 'StrParam(default_dnn_base_path)' 'StrParam('\'''${final.cudaPackages_11_6.cudnn}'\''')'
                    '';

                    # needs to be postFixup so it runs before pythonImportsCheck even when
                    # doCheck = false (meaning preCheck would be disabled)
                    postFixup = ''
                      mkdir -p check-phase
                      export HOME=$(pwd)/check-phase
                    '';
                    #preConfigure = 
                    #''
                    #  export CC=${final.cudaPackages.cudatoolkit.cc}/bin/gcc CXX=${final.cudaPackages.cudatoolkit.cc}/bin/g++
                    #  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${nixpkgs.lib.makeLibraryPath [ final.cudaPackages.cudatoolkit "${final.cudaPackages.cudatoolkit}" ]}"
                    #  export CUDNN_INCLUDE_DIR=${final.cudaPackages.cudnn}/include
                    #''; # enableCuda ${cudatoolkit}/targets/x86_64-linux/lib

                    #  # patchelf --set-rpath "${lib.makeLibraryPath [ stdenv.cc.cc.lib ]}"
                    #preFixup =  ''
                    #  # For some reason pytorch retains a reference to libcuda even if it
                    #  # is explicitly disabled with USE_CUDA=0.
                    #  find $out -name "*.so" -exec ${nixpkgs.patchelf}/bin/patchelf --remove-needed libcuda.so.1 {} \;
                    #'';
                    propagatedBuildInputs = [
                      pyfinal.libgpuarray
                      pyfinal.numpy
                      pyfinal.numpy.blas
                      pyfinal.scipy
                      pyfinal.setuptools
                      #pyfinal.six
                      pyfinal.pandas
                      pyfinal.filelock
                      final.cudaPackages_11_6.cudatoolkit
                      final.cudaPackages_11_6.cudnn
                    ];
                  });

                traitlets = pyprev.traitlets.overridePythonAttrs (old: {
                  propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ pyfinal.hatchling pyfinal.numpy ];
                });

                # jupyterWith also uses six causing six package collision in closure
                #six = pyprev.six.overridePythonAttrs (old: rec {
                #  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ pyprev.setuptools ]) ((old.nativeBuildInputs or [ ]) ++ [
                #  ]);
                #});

                xarray-einstats = pyprev.xarray-einstats.overridePythonAttrs (old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ pyfinal.flit-core ];
                });
                ###############################################################
                #logical-unification = pyprev.buildPythonPackage rec {
                #  pname = "logical-unification";
                #  version = "0.4.5";

                #  propagatedBuildInputs = [ pyfinal.multipledispatch pyfinal.toolz ];

                #  src = pyprev.fetchPypi {
                #    inherit pname version;
                #    sha256 = "sha256-fGpsG3xrqg9bmvk/Bs/I0kGba3kzRrZ47RNnwFznRVg=";
                #  };

                #  doCheck = false;
                #};

                #cons = pyprev.buildPythonPackage rec {
                #  pname = "cons";
                #  version = "0.4.5";

                #  propagatedBuildInputs = [ pyfinal.logical-unification ];

                #  src = pyprev.fetchPypi {
                #    inherit pname version;
                #    sha256 = "sha256-tGtIrbWlr39EN12jRtkm5VoyXU3BK5rdnyAoDTs3Qss=";
                #  };

                #  doCheck = false;
                #};

                #etuples = pyprev.buildPythonPackage rec {
                #  pname = "etuples";
                #  version = "0.3.4";

                #  propagatedBuildInputs = [ pyfinal.cons ];

                #  src = pyprev.fetchPypi {
                #    inherit pname version;
                #    sha256 = "sha256-mAUTeb0oTORi2GjkTDiaIiBrNcSVztJZBBrx8ypUoKM=";
                #  };

                #  doCheck = false;
                #};

                #aesara =
                #  let
                #    blas = final.blas_custom; # not prev.blas
                #    lapack = final.lapack_custom; # not prev.lapack
                #    blasImplementation = nixpkgs.lib.nameFromURL blas.name "-";
                #    cfg = prev.writeTextFile {
                #      name = "site.cfg";
                #      text = (
                #        nixpkgs.lib.generators.toINI
                #          { }
                #          {
                #            ${blasImplementation} = {
                #              include_dirs = "${blas}/include";
                #              library_dirs = "${blas}/lib";
                #            } // nixpkgs.lib.optionalAttrs (blasImplementation == "mkl") {
                #              mkl_libs = "mkl_rt";
                #              lapack_libs = "";
                #            };
                #          }
                #      );
                #    };
                #    wrapped = command: buildTop: buildInputs:
                #      final.runCommandCC "${command}-wrapped" { inherit buildInputs; } ''
                #        type -P '${command}' || { echo '${command}: not found'; exit 1; }
                #        cat > "$out" <<EOF
                #        #!$(type -P bash)
                #        $(declare -xp | sed -e '/^[^=]\+="\('"''${NIX_STORE//\//\\/}"'\|[^\/]\)/!d')
                #        declare -x NIX_BUILD_TOP="${buildTop}"
                #        $(type -P '${command}') "\$@"
                #        EOF
                #        chmod +x "$out"
                #      '';

                #    # Theano spews warnings and disabled flags if the compiler isn't named g++
                #    cxx_compiler_name =
                #      if final.stdenv.cc.isGNU then "g++" else
                #      if final.stdenv.cc.isClang then "clang++" else
                #      throw "Unknown C++ compiler";
                #    cxx_compiler = wrapped cxx_compiler_name "\\$HOME/.theano"
                #      ([ final.cudaPackages_11_6.cudnn final.cudaPackages_11_6.cudatoolkit ]);

                #  in
                #  #pyprev.buildPythonPackage rec {
                #  pyprev.aesara.overrideAttrs (old: {
                #    #$prePhases unpackPhase patchPhase $preConfigurePhases configurePhase $preBuildPhases buildPhase checkPhase $preInstallPhases installPhase fixupPhase installCheckPhase $preDistPhases distPhase $postPhases
                #    #pname = "aesara";
                #    #version = "2.6.6";
                #    postPatch = ''
                #      substituteInPlace aesara/configdefaults.py \
                #        --replace 'StrParam(param, is_valid=warn_cxx)' 'StrParam('\'''${cxx_compiler}'\''', is_valid=warn_cxx)' \
                #        --replace 'rc == 0 and config.cxx != ""' 'config.cxx != ""'
                #      substituteInPlace aesara/configdefaults.py \
                #        --replace 'StrParam(get_cuda_root)' 'StrParam('\'''${final.cudaPackages_11_6.cudatoolkit}'\''')'
                #      substituteInPlace aesara/configdefaults.py \
                #        --replace 'StrParam(default_dnn_base_path)' 'StrParam('\'''${final.cudaPackages_11_6.cudnn}'\''')'
                #    '';

                #    preBuild = ''
                #      ln -s ${cfg} site.cfg
                #    '';

                #    # needs to be postFixup so it runs before pythonImportsCheck even when
                #    # doCheck = false (meaning preCheck would be disabled)
                #    postFixup = ''
                #      mkdir -p check-phase
                #      export HOME=$(pwd)/check-phase
                #    '';

                #    buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [
                #      blas
                #      lapack
                #    ]);
                #    propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [
                #      blas
                #      lapack
                #      pyfinal.mkl-service
                #    #  pyfinal.minikanren #miniKanren
                #    #  pyfinal.cons
                #    #  pyfinal.typing-extensions

                #    #  pyfinal.numpy
                #    #  pyfinal.numpy.blas
                #    #  pyfinal.scipy
                #    #  pyfinal.filelock

                #    #  #pyfinal.libgpuarray
                #    #  pyfinal.setuptools
                #    #  #pyfinal.six
                #    #  pyfinal.pandas
                #    #  pyfinal.jaxlib
                #    #  final.cudaPackages_11_6.cudatoolkit
                #    #  final.cudaPackages_11_6.cudnn
                #    ]);
                #    #src = pyprev.fetchPypi {
                #    #  inherit pname version;
                #    #  sha256 = "sha256-wC7UW/j31NHKrH/8LSC1MN0fJtvtvUNax2ckRuJtGVA=";
                #    #};

                #    #doCheck = false;
                #  });

                ############################################################################################

                # poetry - virtualenv - cython -> ModuleNotFoundError: No module named 'setuptools'
                # pyparsing - setuptools - setuptools-scm -> error: infinite recursion encountered
                #nbformat = pyfinal.python_selected.pkgs.nbformat.override {
                #  inherit (pyfinal)
                #    pytest
                #    ipython_genutils
                #    testpath
                #    jsonschema
                #    jupyter_core
                #    ;
                #};
                nbformat = pyprev.nbformat.overridePythonAttrs (old: {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.flit-core ]);
                });
                setuptools = (pyfinal.python_selected.pkgs.setuptools.overridePythonAttrs (old: {
                  catchConflicts = false;
                  format = "other";
                })).override {
                  inherit (pyfinal)
                    bootstrapped-pip
                    pipInstallHook;
                    #setuptoolsBuildHook
                };
                # With this, skipSetupToolsSCM in mk-poetry-dep.nix is not needed
                setuptools-scm = pyfinal.python_selected.pkgs.setuptools-scm.override {
                  inherit (pyfinal)
                    packaging
                    tomli
                    #typing-extensions
                    setuptools;
                };

                # nbdev <- fastcore, ghapi dependency : stack overflow (possible infinite recursion)
                pip = pyfinal.python_selected.pkgs.pip.override {
                  inherit (pyfinal)
                    bootstrapped-pip
                    mock
                    scripttest
                    virtualenv
                    pretend
                    pytest
                    pip-tools;
                };

                nbdev = pyprev.nbdev.overridePythonAttrs (old: rec {
                  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [
                    pyfinal.twine
                  ]);
                  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.nativeBuildInputs or [ ]) ++ [
                    pyfinal.twine
                  ]);
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [
                    pyfinal.twine
                  ]);
                  propagatedNativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedNativeBuildInputs or [ ]) ++ [
                    pyfinal.twine
                  ]);
                });

                #fastcore = pyfinal.python_selected.pkgs.fastcore.override {
                #  inherit (pyfinal)
                #    ipython
                #    traitlets
                #    mock
                #    pytestCheckHook
                #    nose
                #  ;
                #};
                #ghapi = pyfinal.python_selected.pkgs.ghapi.override {
                #  inherit (pyfinal)
                #    ipython
                #    traitlets
                #    mock
                #    pytestCheckHook
                #    nose
                #  ;
                #};

                # matplotlib : lib.optional should be fixed to lib.optionals
                matplotlib = pyprev.matplotlib.overridePythonAttrs (old: rec {
                  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.nativeBuildInputs or [ ]) ++ [ 
                    pyfinal.setuptools-scm
                    pyfinal.setuptools-scm-git-archive
                  ]);
                });
                contourpy = pyprev.contourpy.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.pybind11 ]);
                });

                # needed by requests needed by twine
                idna = pyprev.idna.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.flit-core ]);
                });

                PyTDC = pyprev.pytdc;

                quarto = pyprev.quarto.override {
                  preferWheel = true;
                };

                rdkit = pyprev.rdkit-pypi;

                jupyter_core = pyfinal.python_selected.pkgs.jupyter_core.override {
                  inherit (pyfinal)
                    ipython
                    traitlets
                    mock
                    pytestCheckHook
                    nose
                  ;
                };
                jupyter-core = pyfinal.jupyter_core; # Not pyprev!!
                termcolor = pyprev.termcolor.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [
                    pyfinal.hatchling
                    # 'hatchling.build' has no attribute 'prepare_metadata_for_build_wheel': needs hatch-vcs not importlib-metadata
                    pyfinal.hatch-vcs
                  ]);
                });

                torch = (pyfinal.python_selected.pkgs.pytorch.override {
                  cudaSupport = true;
                  MPISupport = true;
                  blas = final.blas_custom;
                  cudaPackages = final.cudaPackages_11_6;
                  inherit (final)
                    magma
                    mpi
                  ;
                  inherit (pyfinal)
                    # Native build inputs
                    pybind11
                    # Propagated build inputs
                    numpy
                    pyyaml
                    cffi
                    click
                    typing-extensions
                    # Unit tests
                    hypothesis
                    psutil
                    # dependencies for torch.utils.tensorboard
                    pillow
                    six
                    future
                    tensorboard
                    protobuf
                  ;
                }).overridePythonAttrs (old: rec {
                  USE_SYSTEM_BIND11 = true;
                  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [
                    pyfinal.pybind11
                  ]);
                });
                xgboost = pyfinal.python_selected.pkgs.xgboost.override {
                  inherit (pyfinal)
                    pytestCheckHook
                    scipy
                    scikit-learn
                    pandas
                    matplotlib
                    graphviz
                    datatable
                    hypothesis
                    ;
                };

                # Without tqdm, numpy build error occurs
                #tqdm = final.python3.pkgs.tqdm.overridePythonAttrs (old: rec {
                #  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [ pyprev.numpy ]); #pyfinal.toml 
                #});
              }));
          });
          #polynote = final.stdenv.mkDerivation rec {
          #  pname = "polynote";
          #  version = "0.4.5";
          #  src = builtins.fetchurl {
          #    url = "https://github.com/polynote/polynote/releases/download/0.4.5/polynote-dist.tar.gz";
          #    sha256 = "sha256:1m7braf2d9gmqz270xrj0w4bc3j58bz0mx3h1ik9p1221dz2xc1j"; #prev.lib.fakeSha256;
          #  };
          #  buildInputs = (with python_custom.pkgs; [ virtualenv ipython nbconvert numpy pandas jedi jep jsonschema ]);
          #  patchPhase = ''
          #    substituteInPlace polynote.py \
          #      --replace 'os.path.dirname(os.path.realpath(__file__))' 'os.getcwd()' \
          #      --replace 'p.joinpath("jep") for p in paths if p.joinpath("jep").exists()' '"${python_custom.pkgs.jep}/lib/python3.9/site-packages/jep"' \
          #      --replace 'cmd = f"java' 'cmd = f"${final.openjdk8}/bin/java' \
          #      --replace '-Djava.library.path={jep_path}' '-Djava.library.path=${final.openjdk8}:${final.glibc}/lib:{jep_path}'
          #  '';
          #  installPhase = ''
          #    mkdir -p $out/bin
          #    cp polynote.py $out/bin/polynote
          #    chmod +x $out/bin/polynote
          #  '';
          #};
          python39 = prev.python39.override (old: {
            # for jupyterWith!
            packageOverrides = prev.lib.composeManyExtensions [
              (old.packageOverrides or (_: _: { }))
              (python-final: python-prev: {
                #httplib2 = python-prev.httplib2.overridePythonAttrs (old: {
                #  doCheck = false;
                #});
                numpy = python-prev.numpy.overridePythonAttrs (old:
                  let
                    blas = final.mkl; # not prev.blas
                    blasImplementation = nixpkgs.lib.nameFromURL blas.name "-";
                    cfg = final.writeTextFile {
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
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ final.gfortran ];
                    buildInputs = (old.buildInputs or [ ]) ++ [ blas ];
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
              })
            ];
          });
        })

        #(poetry2nix_nix_community.overlay)

        #(final: pyprev: {
        #  # The application
        #  myapp = pyprev.poetry2nix.mkPoetryApplication {
        #    projectDir = ./.;
        #  };
        #})
        #jupyterWith.overlays.jupyterWith
        #jupyterWith.overlays.haskell
        #jupyterWith.overlays.python
      ] ++ (builtins.attrValues jupyterWith.overlays)
        #++ [ dpaetzel.pymc4 ] 
        ++ [
        (final: prev: {
          jupyterWith_python_custom = final.jupyterWith.override {
            python3 = self.python_custom.x86_64-linux;
            #python3 = self.python_custom.x86_64-linux.pkgs.python39.pkgs;
          };
        })
        ]
      );
    } // (flake-utils.lib.eachSystem [ "x86_64-linux" "x86_64-darwin" ] (system: # appending behind output
      rec
      {
        pkgs = import nixpkgs {
          inherit system; #system = "x86_64-linux";
          config.allowUnfree = true;
          config.cudaSupport = true;
          overlays = [ self.overlay ];
          #overlays = (builtins.attrValues jupyterWith.overlays) ++ [ self.overlay ]; # [ (import ./haskell-overlay.nix) ];
        };

        python_custom = pkgs.poetry2nix.mkPoetryEnv rec {
          projectDir = ./.;
          python = pkgs.python39;
          editablePackageSources = {
            mypackages = ~/Jupyter_Python; #./.; not working
          };
          #extraPackages = ps: [ ps.pytorch_custom2 ];
          #editablePackageSources = {
          #  ronald_bdl = "${builtins.getEnv "HOME"}/MS-Thesis/my-python-package/ronald_bdl";
          #ronald_bdl = ./my-python-package/ronald_bdl;
          #};

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
        #  numpy
        #])).override (args: { ignoreCollisions = true; });

        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
        depNames = builtins.attrNames pyproject.tool.poetry.dependencies;

        iPythonWithPackages = pkgs.jupyterWith_python_custom.kernels.iPythonWith {
          name = "ms-thesis--env";
          python3 = python_custom;
          packages = p:
            let
              # Building the local package using the standard way.
              myPythonPackage = p.buildPythonPackage {
                pname = "MyPythonPackage";
                version = "1.0";
                src = ./my-python-package;
              };
              #myPythonPackage = p.buildPythonPackage {
              #  pname = "nbdev-cards";
              #  version = "0.0.1";
              #  src = ./nbdev_cards;
              #  buildInputs = [
              #    p.fastcore
              #  ];
              #};
              # Getting dependencies using Poetry.
              poetryDeps =
                builtins.map (name: builtins.getAttr name p) depNames;
              # p : gets packages from 'python3 = python' ? maybe?
            in
            #[  ] ++ # adds nixpkgs.url version  python pkgs.
            [ myPythonPackage ] ++ poetryDeps; ### ++ (poetryExtraDeps p);
        };
        jupyterEnvironment = (pkgs.jupyterWith_python_custom.jupyterlabWith {
          kernels = [ iPythonWithPackages ];
          extraPackages = ps: [
            ps.protobuf # ps.protobuf3_9
          ];
          #extraJupyterPath = pkgs: [
          #export CUDA_PATH=${pkgs.cudatoolkit}
          #export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
          #export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
          #export EXTRA_CCFLAGS="-I/usr/include"
          #];
        });
        #nixGL = fetchzip { fetchTarball {
        #  url = "https://github.com/guibou/nixGL/archive/master.tar.gz";
        #  sha256 = "sha256:093lf41pp22ndkibm1fqvi78vfzw255i3313l72dwkk86q9wsbzr";
        #pli#b.fakeSha256;
        #};
        #myNixGL = (import "${nixGL}/default.nix" {
        #  nvidiaVersion = "440.64.00";
        #  nvidiaHash = "08yln39l82fi5hmn06xxi3sl6zb4fgshhhq3m42ksiglradjd0ah";
        #  inherit pkgs;
        #}).nixGLNvidia;
        apps.jupyter-lab = {
          type = "app";
          program = "${jupyterEnvironment}/bin/jupyter-lab";
        };
        apps.jupyter-notebook = {
          type = "app";
          program = "${python_custom.pkgs.notebook}/bin/jupyter-notebook";
        };
        apps.jupyter-notebookk = {
          type = "app";
          program = "${python_custom.pkgs.notebook}/bin/jupyter-notebook";
        };
        packages.default = packages.poetry;
        packages.poetry = python_custom.pkgs.poetry;
        #packages.jupyterlab = jupyterEnvironment;
        packages.nbdev = python_custom.pkgs.nbdev;
          #source ${pkgs.writeShellScriptBin "export" ''
          #  ''}/bin/export
          #exec ${python_custom.pkgs.nbdev}/bin/nbdev_preview --port 3334
        #packages.polynote = pkgs.polynote;
        #packages.jep = pkgs.python3.pkgs.jep;
        #packages.pythonenv = python_custom;
        
        packages.quarto = with builtins; with nixpkgs; with pkgs; stdenv.mkDerivation rec {
          pname = "quarto";
          version = "1.1.189";

          src = fetchurl {
            url = "https://github.com/quarto-dev/quarto-cli/releases/download/v${version}/quarto-${version}-linux-amd64.tar.gz";
            sha256 = "1a3xsgqdccm4ky1xjnin1idpp8gsansskq37c00mrxz1raxn1mi7";
          };

          nativeBuildInputs = [
            makeWrapper
          ];

          patches = [
              ./fix-deno-path.patch
          ];

          preFixup = ''
            wrapProgram $out/bin/quarto \
              --prefix PATH : ${lib.makeBinPath [ deno ]} \
              --prefix QUARTO_PANDOC : ${lib.makeBinPath [ pandoc ]}/pandoc \
              --prefix QUARTO_ESBUILD : ${lib.makeBinPath [ esbuild ]}/esbuild \
              --prefix QUARTO_DART_SASS : ${lib.makeBinPath [ nodePackages.sass ]}/sass \
              --prefix QUARTO_R : ${lib.makeBinPath [ (rWrapper.override { packages = [ rPackages.rmarkdown ]; }) ]}/R
          '';

          installPhase = ''
            runHook preInstall

            mkdir -p $out/{bin,share}
            rm -r bin/tools
            mv bin/vendor/import_map.json bin
            rm -r bin/vendor

            mv bin/* $out/bin
            mv share/* $out/share

            runHook postInstall
          '';
        };
        #nativeBuildInputs = [
        #  pkgs.dpkg
        #  pkgs.autoPatchelfHook
        #];
        #unpackCmd = "dpkg-deb -x $curSrc .";
        
        #devShells.quarto = pkgs.mkShell rec {
        #  packages = [
        #    self.packages.${system}.quarto-cli
        #  ];

        #  #JULIA_DEPOT_PATH = "./.julia_depot";

        #  shellHook = ''export LD_LIBRARY_PATH='' + LD_LIBRARY_PATH + ''
        #    export TF_ENABLE_ONEDNN_OPTS=0 # when using GPU, oneDNN off is recommended 
        #  '';
        #};
        #devShells.nbdev = pkgs.mkShell rec {
        #  packages = [
        #    jupyterEnvironment
        #    python_custom.pkgs.nbdev
        #  ];

        #  #JULIA_DEPOT_PATH = "./.julia_depot";

        #  shellHook = ''export LD_LIBRARY_PATH='' + LD_LIBRARY_PATH + ''
        #    export TF_ENABLE_ONEDNN_OPTS=0 # when using GPU, oneDNN off is recommended 
        #  '';
        #};

        devShells.default = pkgs.mkShell rec {
          packages = [
            # stdenv reference :
            # https://discourse.nixos.org/t/nixos-with-poetry-installed-pandas-libstdc-so-6-cannot-open-shared-object-file/8442/3
            # https://nixos.wiki/wiki/Packaging/Quirks_and_Caveats#ImportError:_libstdc.2B.2B.so.6:_cannot_open_shared_object_file:_No_such_file
            #stdenv.cc.cc.lib
            python_custom
            #iPythonWithPackages.runtimePackages 
            #jupyterEnvironment
            #python_custom.pkgs.poetry
            python_custom.pkgs.nbdev
            #python_custom.pkgs.quarto
            #python_custom.pkgs.kaggle
            pkgs.graphviz

            self.packages.${system}.quarto

            nixgl.defaultPackage.${system}
            pkgs.linuxPackages.nvidia_x11

            pkgs.nvtop
            pkgs.deno
            pkgs.lldb
            # polynote : go to polynote folder that has deps, notebooks folder and config.yml inside.
            # run 'polynote'. current port in config.yml is 5555
            #pkgs.polynote
            #(pkgs.lib.getBin pkgs.graphviz)
            #(pkgs.lib.getBin pkgs.caffe)
            #(pkgs.lib.getBin python_custom)

            #iJulia.runtimePackages
          ];

          #JULIA_DEPOT_PATH = "./.julia_depot";

          shellHook = ''export LD_LIBRARY_PATH='' + LD_LIBRARY_PATH + ''
            export TF_ENABLE_ONEDNN_OPTS=0 # when using GPU, oneDNN off is recommended 
            export PYTHONPATH=~/Jupyter_Python/nbdev_cards:$PYTHONPATH
          '';
          #trivial = nixpkgs.lib.concatStringsSep "/" [(builtins.getEnv "HOME") "Jupyter_Python" "src"];
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
#
#$prePhases unpackPhase patchPhase $preConfigurePhases configurePhase $preBuildPhases buildPhase checkPhase $preInstallPhases installPhase fixupPhase installCheckPhase $preDistPhases distPhase $postPhases
#
# Library Path
#   - old version : 
#   export LD_LIBRARY_PATH=${pkgs.cudaPackages_11_6.cudatoolkit}/lib64:${pkgs.cudaPackages_11_6.cudatoolkit.lib}/lib:${pkgs.cudaPackages_11_6.cudnn}/lib:${pkgs.nvidia_custom}/lib:${nixpkgs.lib.makeLibraryPath [ pkgs.cudaPackages_11_6.cudatoolkit "${pkgs.cudaPackages_11_6.cudatoolkit}" ]}:$LD_LIBRARY_PATH
#   - new version :
#   export LD_LIBRARY_PATH=${nixpkgs.lib.makeLibraryPath [ pkgs.cudaPackages_11_6.cudatoolkit "${pkgs.cudaPackages_11_6.cudatoolkit}" pkgs.cudaPackages_11_6.cudnn pkgs.nvidia_custom ]}:$LD_LIBRARY_PATH
# ${nixpkgs.lib.makeLibraryPath [ pkgs.cudaPackages_11_6.cudatoolkit ]} == ${pkgs.cudaPackages_11_6.cudatoolkit.lib}/lib
# ${nixpkgs.lib.makeLibraryPath [ "${pkgs.cudaPackages_11_6.cudatoolkit}" ]} == ${pkgs.cudaPackages_11_6.cudatoolkit}/lib
# 
# Error Log
#
# - numpy error
# error: builder for '/nix/store/4w7y08bmrz1acs3xxz7jmdqrvr0cgcg2-python3.9-numpy-1.21.5.drv' failed with exit code 1;
#        last 10 log lines:
#        >   Requested   : 'min'
#        >   Enabled     : SSE SSE2 SSE3
#        >   Flags       : -msse -msse2 -msse3
#        >   Extra checks: none
#        >
#        > CPU dispatch  :
#        >   Requested   : 'max -xop -fma4'
#        >   Enabled     : SSSE3 SSE41 POPCNT SSE42 AVX F16C FMA3 AVX2 AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL
#        >   Generated   : none
#        > CCompilerOpt.cache_flush[809] : write cache to path -> /tmp/nix-build-python3.9-numpy-1.21.5.drv-0/numpy-1.21.5/build/temp.linux-x86_64-3.9/ccompiler_opt_cache_clib.py
#        For full logs, run 'nix log /nix/store/4w7y08bmrz1acs3xxz7jmdqrvr0cgcg2-python3.9-numpy-1.21.5.drv'.
# 
# Solution : add pyself.numpy to nativeBuildInputs
#                tqdm = pysuper.tqdm.overridePythonAttrs (old: rec {
#                  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [  ]) ((old.nativeBuildInputs or [ ]) ++ [ pyself.numpy ]);
#                });
#
# - Module not found error
#   - Check whether file structure has changed
#     - ex) poetry/__init__.py -> src/poetry/__init__.py -> Erased
#
# - When flake.lock is not updated
# poetry cache clear pypi --all
# 
# - Updating src.json
# nix-prefetch-url --unpack https://github.com/sepiabrown/poetry/archive/refs/heads/download_fix.tar.gz
#
# - infinite recursion
# Packages that create '[[package]] name = "setuptools"' item in poetry.lock causes infinite recursion.
# Solution : 
# - Erase setuptools item in poetry.lock 
# - Override setuptools with pkgs outside poetry2nix
#   - poetry itself needs setuptools of poetry2nix so poetry needs to be overriden too
#   - If setuptools-scm is also overriden, skipSetupToolsSCM in mk-poetry-dep.nix is not needed
#
# - packages that are not yet overriden in poetry2nix
# ex) xgboost
# copying xgboost/LICENSE -> build/temp.linux-x86_64-3.9/xgboost                                                  
# INFO:XGBoost build_ext:Building from source. /tmp/nix-build-python3.9-xgboost-1.6.1.drv-0/lib/libxgboost.so                                 
# INFO:XGBoost build_ext:Run CMake command: ['cmake', 'xgboost', '-GUnix Makefiles', '-DUSE_OPENMP=1', '-DUSE_CUDA=0', '-DUSE_NCCL=0', '-DBUILD_WITH_SHARED_NCCL=0', '-DHIDE_CXX
#HDFS=0', '-DUSE_AZURE=0', '-DUSE_S3=0', '-DPLUGIN_DENSE_PARSER=0']       
# error: [Errno 2] No such file or directory: 'cmake'                                                                                    
# error: subprocess-exited-with-error                                                                                                  
# × python setup.py bdist_wheel did not run successfully.                                                                                   
# │ exit code: 1                                                                                             
# ╰─> See above for output.                                                                                                 
# note: This error originates from a subprocess, and is likely not a problem with pip.
#
# Solution :
# - Overriden with 
#   xgboost = pyfinal.python_selected.pkgs.xgboost;
#
# - numpy gets built again even if we already have numpy built
#   - due to package update, the associated package might have started following numpy
#
# Solution :
# - Execute 'nix develop --print-build-logs' and find the .drv
# - Execute 'nix-store -q --referrers-closure /nix/store/~' to find out packages that are referencing ~
# - Usually, packages that are overriden in flake from nixpkgs.
#
# - collision between packages
# error: collision between `/nix/store/9m1idzg5lbx8y9njrry1jkaybwjz42xm-python3.9-tensorboard-2.9.1/bin/.tensorboard-wrapped'
# and `/nix/store/fk2h94blk65pwsj2058034z01ic22ggx-python3.9-tensorflow-gpu-2.9.1/bin/.tensorboard-wrapped'
# 
# Solution :
# - `tensorboard` was included in `propagatedBuildInputs` of `tensorflow-gpu`.
#   Move `tensorboard` to `buildInputs` of `tensorflow-gpu`
#
# - Duplicated package error 1
# pythonCatchConflictsPhase                                                                                                               
# Found duplicated packages in closure for dependency 'attrs':                                                                            
#   attrs 21.4.0 (/nix/store/9bkjgskj10vaw86j391c66yj0f3dl6ki-python3.9-attrs-21.4.0/lib/python3.9/site-packages)                         
#   attrs 21.4.0 (/nix/store/a4kd5fvqnqcb1ab79s74d46fns1y866d-python3.9-attrs-21.4.0/lib/python3.9/site-packages)                         
# Found duplicated packages in closure for dependency 'packaging':                                                                        
#   packaging 21.3 (/nix/store/j5zjc544xnwhq72ahqcl2hmbnrqm94jj-python3.9-packaging-21.3/lib/python3.9/site-packages)                     
#   packaging 21.3 (/nix/store/p5j7ysv9kmn9jwa1n3azmg1vqbd894yc-python3.9-packaging-21.3/lib/python3.9/site-packages)                     
# Found duplicated packages in closure for dependency 'pyparsing':                                                                        
#   pyparsing 3.0.9 (/nix/store/ppqwzdz4rcz15ack4a9wwjq07sv2l0mg-python3.9-pyparsing-3.0.9/lib/python3.9/site-packages)                   
#   pyparsing 3.0.9 (/nix/store/f2218vyr4xmbpz27vixdd5s6a33igp3m-python3.9-pyparsing-3.0.9/lib/python3.9/site-packages)                   
# Found duplicated packages in closure for dependency 'tomli':                                                                            
#   tomli 2.0.1 (/nix/store/aqkwcr0afnjasj9mk1b8r8bhz8ys56hq-python3.9-tomli-2.0.1/lib/python3.9/site-packages)                           
#   tomli 2.0.1 (/nix/store/jnfx8vzalnyzcq2vd12w1bcg71h9n8hm-python3.9-tomli-2.0.1/lib/python3.9/site-packages)                           
#
# Package duplicates found in closure, see above. Usually this happens if two packages depend on different version of the same dependency.
# For full logs, run 'nix log /nix/store/dc3izhhs0fx603gnnmsn7cqyvjqz6nrs-python3.9-xgboost-1.5.2.drv'.
#
# Solution :
# - Execute `nix-tree /nix/store/dc3izhhs0fx603gnnmsn7cqyvjqz6nrs-python3.9-xgboost-1.5.2.drv --derivation`
# - search duplicated packages and find the parents packages that are causing the issue.
# - This time, it was `pytest-check-hook.drv` from `python39.pkgs.pytestCheckHook`.
# - In search.nixos.org, `python39Packages.xgboost` didn't have `pytestCheckHook` for an argument.
# - It turns out that the search result was directing `python39Packages.xgboost` to `xgboost`.
# - From hound, `python39Packages.xgboost` had `pytestCheckHook` as an argument
# - Adding `pytestCheckHook` to `xgboost` override solved every problems.
#
# - Duplicated package error 2
# pythonCatchConflictsPhase                                                                                                               
# Found duplicated packages in closure for dependency 'jupyter-core':                                                                     
#   jupyter-core 4.9.2 (/nix/store/mwdw6m14n2p5wp5yd6xcxmnf9nwx71nc-python3.9-jupyter_core-4.9.2/lib/python3.9/site-packages)             
#   jupyter-core 4.11.1 (/nix/store/rx4ih1vg5r4wxdl17jnf5jiibw9953ir-python3.9-jupyter-core-4.11.1/lib/python3.9/site-packages)           
#                                                                                                                                         
# Package duplicates found in closure, see above. Usually this happens if two packages depend on different version of the same dependency.
# For full logs, run 'nix log /nix/store/iavdqn904phbwzspd22daja8h44ywlp3-python3.9-notebook-6.4.12.drv'.                                 
#
# Solution :
# - Executed `nix-tree` and found that `notebook` depends on `jupyter_core`(A) and `nbformat`.
# - But `nbformat` depends on different `jupyter_core`(B) version.
# - Tried overriding `nbformat` but resulted in more duplicated packages with `jupyter_core`(A).
# - Thus, overriden `jupyter_core`(A)
#
# - Duplicated package error 3
# pythonCatchConflictsPhase                                                                                                               
# Found duplicated packages in closure for dependency 'jupyter-core':
#   jupyter-core 4.9.2 (/nix/store/g85wjw10di8h8y9lqhian5jjzcymy960-python3.9-jupyter_core-4.9.2/lib/python3.9/site-packages)
#   jupyter-core 4.9.2 (/nix/store/mwdw6m14n2p5wp5yd6xcxmnf9nwx71nc-python3.9-jupyter_core-4.9.2/lib/python3.9/site-packages)
#
# Package duplicates found in closure, see above. Usually this happens if two packages depend on different version of the same dependency.
# For full logs, run 'nix log /nix/store/jzlgrmm30fn5z82flfma09311y51vymz-python3.9-notebook-6.4.12.drv'.
#
# Solution :
# - `jupyter-core` and `jupyter_core` were recognised as a different entity 
# - Set `jupyter-core = pyfinal.jupyter_core`
#
# error: stack overflow (possible infinite recursion)
#
# Solution :
# - Use `pyfinal.python_selected.pkgs.XXXX.override`
# - Examples : setuptools, setuptools-scm, pip
#
# Error: No module named 'pkg_resources' when running nbdev inside devShell
#
# Solution :
# - We need both `python_custom` and `python_custom.pkgs.nbdev` listed in devShell
