{
  description = "fun with jupyterWith and poetry2nix";
  inputs = {
    nixpkgs.url = "github:sepiabrown/nixpkgs/pythonRelaxDepsHook"; 
    nixpkgs_2211.url = "github:sepiabrown/nixpkgs/nixos-unstable_pythonRelaxDepsHook"; 
    #nixpkgs.url = "nixpkgs/nixos-22.05"; 
    #nixpkgs_2211.url = "nixpkgs/nixos-unstable"; 
    poetry2nix = {
      #url = "github:sepiabrown/poetry2nix/python_overlay_fix";
      url = "github:nix-community/poetry2nix";
      #url = "github:nix-community/poetry2nix?ref=refs/pull/787/merge";
      # https://github.com/nix-community/poetry2nix/issues/810#issuecomment-1312531883
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
    jupyterWith = {
      url = "github:tweag/jupyterWith/deaa6c66165fd1ebe8617a8f133ad45110ac659c"; 
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs: with inputs; 
    let
      LD_LIBRARY_PATH = ''
        ${nixpkgs.lib.makeLibraryPath [ self.pkgs.x86_64-linux.cudaPackages.cudatoolkit "${self.pkgs.x86_64-linux.cudaPackages.cudatoolkit}" self.pkgs.x86_64-linux.cudaPackages.cudnn self.pkgs.x86_64-linux.nvidia_custom ]}:$LD_LIBRARY_PATH
      '';

      pkgs_2211 = import nixpkgs_2211 {
        system = "x86_64-linux";
        config.allowUnfree = true;
        config.cudaSupport = true;
        config.mklSupport = true;
        #overlays = [ self.overlay ];
        #overlays = (builtins.attrValues jupyterWith.overlays) ++ [ self.overlay ]; # [ (import ./haskell-overlay.nix) ];
      };
    in
    {
      # Order of overlays matter! Earlier ones get overriden before later ones.
      overlay = nixpkgs.lib.composeManyExtensions (
        #(builtins.attrValues jupyterWith.overlays) ++ 
        [
        #(final: prev: {
        #  poetry2nix = (poetry2nix.overlay final prev).poetry2nix.overrideScope' (p2nixfinal: p2nixprev: {
        #    # pyfinal & pyprev refers to python packages
        #    defaultPoetryOverrides = (p2nixprev.defaultPoetryOverrides.overrideOverlay (pyfinal: pyprev: {
        #      tensorflow = null;
        #      tensorflow-gpu = null;
        #    }));
        #  });
        #})
        (final: prev: {
          poetry2nix = prev.poetry2nix.overrideScope' (p2nixfinal: p2nixprev: {
            # pyfinal & pyprev refers to python packages
            defaultPoetryOverrides = (p2nixprev.defaultPoetryOverrides.overrideOverlay (pyfinal: pyprev: {
              babel = null;
              Babel = null;
              babel_ = pyprev.babel;

              keras = null;
              Keras = null;
              keras_ = pyprev.keras;
              tensorflow = null;
              tensorflow-gpu = null;
            })).extend (pyfinal: pyprev: {
              babel = pyprev.babel_;
              keras = pyprev.keras_;
            });
          });
        })

        #(final: prev: {
        #  poetry2nix = prev.poetry2nix.overrideScope' (p2nixfinal: p2nixprev: {
        #    # pyfinal & pyprev refers to python packages
        #    defaultPoetryOverrides = (p2nixprev.defaultPoetryOverrides.extend (pyfinal: pyprev: {
        #      babel = pyprev.babel_;
        #      keras = pyprev.keras_;
        #    }));
        #  });
        #})

        (final: prev: {
          blas_custom = (prev.blas.override {
            blasProvider = final.mkl;
          }).overrideAttrs (oldAttrs: {
            buildInputs = (oldAttrs.buildInputs or [ ])
            ++ final.lib.optional final.stdenv.hostPlatform.isDarwin final.fixDarwinDylibNames;
          });

          cudaPackages = prev.cudaPackages_11_6;
          #cudaPackages = prev.cudaPackages_11_3;

          lapack_custom = (prev.lapack.override {
            lapackProvider = final.mkl;
          }).overrideAttrs (oldAttrs: {
            buildInputs = (oldAttrs.buildInputs or [ ])
            ++ final.lib.optional final.stdenv.hostPlatform.isDarwin final.fixDarwinDylibNames;
          });
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
                  let
                    inherit (final.cudaPackages) cudatoolkit cudnn nccl;
  cudatoolkit_joined = final.symlinkJoin {
    name = "${cudatoolkit.name}-unsplit";
    # nccl is here purely for semantic grouping it could be moved to nativeBuildInputs
    paths = [ cudatoolkit.out cudatoolkit.lib nccl.dev nccl.out ];
  };

  brokenArchs = [ "3.0" ]; # this variable is only used as documentation.

  cudaCapabilities = rec {
    cuda9 = [
      "3.5"
      "5.0"
      "5.2"
      "6.0"
      "6.1"
      "7.0"
      "7.0+PTX"  # I am getting a "undefined architecture compute_75" on cuda 9
                 # which leads me to believe this is the final cuda-9-compatible architecture.
    ];

    cuda10 = cuda9 ++ [
      "7.5"
      "7.5+PTX"  # < most recent architecture as of cudatoolkit_10_0 and pytorch-1.2.0
    ];

    cuda11 = cuda10 ++ [
      "8.0"
      "8.0+PTX"  # < CUDA toolkit 11.0
      "8.6"
      "8.6+PTX"  # < CUDA toolkit 11.1
    ];
  };
  final_cudaArchList =
    if !final.cudaSupport || final.cudaArchList != null
    then final.cudaArchList
    else cudaCapabilities."cuda${nixpkgs.lib.versions.major cudatoolkit.version}";

  # Normally libcuda.so.1 is provided at runtime by nvidia-x11 via
  # LD_LIBRARY_PATH=/run/opengl-driver/lib.  We only use the stub
  # libcuda.so from cudatoolkit for running tests, so that we don’t have
  # to recompile pytorch on every update to nvidia-x11 or the kernel.
  cudaStub = final.linkFarm "cuda-stub" [{
    name = "libcuda.so.1";
    path = "${cudatoolkit}/lib/stubs/libcuda.so";
  }];
  cudaStubEnv = nixpkgs.lib.optionalString cudaSupport
    "LD_LIBRARY_PATH=${cudaStub}\${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH ";
              in
              {
                ############################################################
                ### Necessities for solving infinite recursion ###
                python_selected = prev.python39Packages;

                #setuptools = (pyfinal.python_selected.setuptools.overridePythonAttrs (old: {
                setuptools = (pkgs_2211.python39Packages.setuptools.override {
                # 65.3.0 : nixos-22.11
                  inherit (pyfinal)
                    python
                    bootstrapped-pip
                    pipInstallHook
                    setuptoolsBuildHook
                  ;
                }).overridePythonAttrs (old: rec {
                  #catchConflicts = false;
                  #format = "other";
                });

                # With this, skipSetupToolsSCM in mk-poetry-dep.nix is not needed
                #setuptools-scm = pyfinal.python_selected.setuptools-scm.override {
                setuptools-scm = pkgs_2211.python39Packages.setuptools-scm.override {
                # 7.0.5 : nixos-22.11
                  inherit (pyfinal)
                    packaging
                    typing-extensions
                    tomli
                    setuptools;
                };

                #pip = pyfinal.python_selected.pip.override {
                pip = pkgs_2211.python39Packages.pip.override {
                # 22.2.2 : nixos-22.11
                  inherit (pyfinal)
                    bootstrapped-pip
                    mock
                    scripttest
                    virtualenv
                    pretend
                    pytest
                    pip-tools
                  ;
                };
                ### Necessities for solving infinite recursion (end) ###
                ############################################################

                ############################################################
                ### Override python packages if they lack dependencies

                #apache-beam = pyprev.apache-beam.overridePythonAttrs (old: rec {
                #apache-beam = pyfinal.python_selected.apache-beam.overridePythonAttrs (old: {
                #apache-beam = (pkgs_2211.python39Packages.apache-beam.override {
                #  inherit (pyfinal)
                #    cloudpickle
                #    crcmod
                #    cython
                #    dill
                #    fastavro
                #    freezegun
                #    grpcio
                #    grpcio-tools
                #    hdfs
                #    httplib2
                #    mock
                #    mypy-protobuf
                #    numpy
                #    oauth2client
                #    orjson
                #    pandas
                #    parameterized
                #    proto-plus
                #    protobuf
                #    psycopg2
                #    pyarrow
                #    pydot
                #    pyhamcrest
                #    pymongo
                #    pytestCheckHook
                #    python-dateutil
                #    pythonRelaxDepsHook
                #    pytz
                #    pyyaml
                #    requests
                #    requests-mock
                #    scikit-learn
                #    sqlalchemy
                #    tenacity
                #    #testcontainers # doesn't exist in 22.05
                #    typing-extensions;
                #}).overridePythonAttrs (old: {
                #  version = pyprev.apache-beam.version;
                #  src = pyprev.apache-beam.src;
                #});

                cryptography = pyprev.cryptography.overridePythonAttrs (old: rec {
                  #buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [ final. ]);
                  #nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.nativeBuildInputs or [ ]) ++ [ final. ]);
                  #propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal. ]);
                  cargoDeps =
                    prev.rustPlatform.fetchCargoTarball {
                      src = old.src;
                      sourceRoot = "${old.pname}-${old.version}/${cargoRoot}";
                      name = "${old.pname}-${old.version}";
                      #inherit sha256;
                      #sha256 = "sha256-lzHLW1N4hZj+nn08NZiPVM/X+SEcIsuZDjEOy0OOkSc=";
                      sha256 = "sha256-BN0kOblUwgHj5QBf52RY2Jx0nBn03lwoN1O5PEohbwY=";
                    };
                  cargoRoot = "src/rust";
                });
                # = (pyfinal.python_selected..override {
                # = (pkgs_2211.python39Packages..override {
                ## 1.23.0 : nixos-22.05< <22.11
                #  inherit (pyfinal)
                #  ;
                #}).overridePythonAttrs (old: rec {
                #  version = pyprev..version;
                #  src = pyprev..src;
                #});

                comm = pyprev.comm.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.hatchling ]);
                });

                contourpy = pyprev.contourpy.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.pybind11 ]);
                });

                #cython = pyprev.cython.overridePythonAttrs (old: rec {
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [  ]) ++ [ pyfinal.setuptools ]);
                #});
                # = (pyfinal.python_selected..override {
                cython = (pkgs_2211.python39Packages.cython.override {
                # 0.29.32 : nixos-22.11
                  inherit (pyfinal)
                    python
                    numpy
                  ;
                }).overridePythonAttrs (old: rec {
                  #version = pyprev.cython.version; # this overrides with poetry2nix version, reverting our override
                  #src = pyprev.cython.src;
                });

                Cython = pyfinal.cython;

                #dm-sonnet = (pyprev.dm-sonnet.override {
                dm-sonnet = (pkgs_2211.python39Packages.dm-sonnet.override {
                # 2.0.0 : nixos-22.11
                  inherit (pyfinal)
                    absl-py
                    dm-tree
                    docutils
                    numpy
                    tabulate
                    tensorflow-datasets
                    wrapt;
                  tensorflow = pyfinal.tensorflow-gpu;
                }).overridePythonAttrs (old: {
                  propagatedBuildInputs = [
                    pyfinal.dm-tree
                    pkgs_2211.python39Packages.etils
                    pyfinal.numpy
                    pyfinal.tabulate
                    pyfinal.wrapt

                    pyfinal.absl-py
                    pyfinal.six

                    pyfinal.importlib-resources
                    pyfinal.typing-extensions
                    pyfinal.zipp
                  ];
                  #version = pyprev.dm-sonnet.version;
                  #src = pyprev.dm-sonnet.src;
                });

                #dm-tree = pyfinal.python_selected.dm-tree.override {
                #dm-tree = pyprev.dm-tree.override {
                dm-tree = (pkgs_2211.python39Packages.dm-tree.override {
                # 0.1.7 : nixos-22.11
                  inherit (pyfinal)
                    absl-py
                    attrs
                    numpy
                    pybind11
                    wrapt;
                }).overridePythonAttrs (old: {
                  version = pyprev.dm-tree.version;
                  src = pyprev.dm-tree.src;
                });

                #grpcio-tools = pkgs_2211.python39Packages.grpcio-tools.override {
                #  inherit (pyfinal)
                #    protobuf
                #    grpcio;
                #};

                # = (pyfinal.python_selected..override {
                hatch-nodejs-version = (pkgs_2211.python39Packages.hatch-nodejs-version.override {
                # 0.3.0 : nixos-22.11<
                  inherit (pyfinal)
                    pytestCheckHook
                    hatchling
                  ;
                }).overridePythonAttrs (old: rec {
                  #version = pyprev.hatch-nodejs-version.version; # doesn't exist in 22.05
                  #src = pyprev.hatch-nodejs-version.src;
                });

                h5py = (pyfinal.python_selected.h5py.override {
                # 3.6.0 : nixos-22.05
                  inherit (pyfinal)
                    numpy
                    cython
                    six
                    pkgconfig
                    unittest2
                    mpi4py
                    openssh
                    pytestCheckHook
                    cached-property
                  ;
                }).overridePythonAttrs (old: rec {
                  #version = pyprev.h5py.version; # format issue
                  #src = pyprev.h5py.src;
                });

                idna = pyprev.idna.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.flit-core ]);
                });

                ####
                #jax = (pyprev.jax.overridePythonAttrs (old: rec {
                #jax = (pyfinal.python_selected.jax.override {
                jax = (pkgs_2211.python39Packages.jax.override {
                # 0.3.23 : nixos-22.11
                  inherit (pyfinal)
                    absl-py
                    jaxlib
                    matplotlib
                    numpy
                    opt-einsum
                    pytestCheckHook
                    pytest-xdist # duplicated packages error
                    scipy
                    typing-extensions
                    ;
                  etils = pkgs_2211.python39Packages.etils;
                  blas = final.blas_custom;
                  lapack = final.lapack_custom;
                }).overridePythonAttrs (old: rec {
                  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [
                    #pyfinal.jaxlib
                  ]);
                  propagatedBuildInputs = [
                    pyfinal.absl-py
                    pkgs_2211.python39Packages.etils
                    pyfinal.numpy
                    pyfinal.opt-einsum
                    pyfinal.scipy
                    pyfinal.typing-extensions

                    pyfinal.importlib-resources
                    pyfinal.zipp
                  ];
                  #version = pyprev.jax.version; # format issue
                  #src = pyprev.jax.src;
                });

                ####
                #jaxlib = (pkgs_2211.python39Packages.jaxlib.override {
                ##jaxlib = pyprev.jaxlib.override {
                jaxlib = (pyfinal.python_selected.jaxlib.override {
                # 0.3.20 : nixos-22.05< <nixos-22.11
                  inherit (pyfinal)
                    # Build-time dependencies:
                    python
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
                    ;
                  cudaSupport = true;
                  cudaPackages = final.cudaPackages;
                  mklSupport = true;
                }).overridePythonAttrs (old: rec {
                  version = pyprev.jaxlib.version;
                  src = pyprev.jaxlib.src;
                });

                # = (pyfinal.python_selected..override {
                jsonschema = (pkgs_2211.python39Packages.jsonschema.override {
                # 4.17.1 : nixos-22.11<
                  inherit (pyfinal)
                    attrs
                    hatch-fancy-pypi-readme
                    hatch-vcs
                    hatchling
                    importlib-metadata
                    importlib-resources
                    pyrsistent
                    twisted
                    typing-extensions
                  ;
                }).overridePythonAttrs (old: rec {
                  version = pyprev.jsonschema.version;
                  src = pyprev.jsonschema.src;
                });

                # = (pyfinal.python_selected..override {
                jupyter = (pkgs_2211.python39Packages.jupyter.override {
                # 1.0.0 : nixos-22.11<
                  inherit (pyfinal)
                    notebook
                    qtconsole
                    jupyter_console
                    nbconvert
                    ipykernel
                    ipywidgets
                  ;
                }).overridePythonAttrs (old: rec {
                  #version = pyprev.jupyter.version;
                  #src = pyprev.jupyter.src;
                });

                jupyterlab = pyprev.jupyterlab.overridePythonAttrs (oldAttrs: {
                  makeWrapperArgs = (oldAttrs.makeWrapperArgs or [ ]) ++ [
                    "--set LD_LIBRARY_PATH"
                    LD_LIBRARY_PATH
                    "--set TF_ENABLE_ONEDNN_OPTS 0" # when using GPU, oneDNN off is recommended 
                    "--set XLA_FLAGS --xla_gpu_cuda_data_dir=${cudatoolkit}"
                    #"--set AESARA_FLAGS device=cuda0,dnn__base_path=${cudnn},blas__ldflags=-lblas,dnn__library_path=${cudnn}/lib,dnn__include_path=${cudnn}/include"#${nixpkgs.lib.makeLibraryPath [ cudnn ]}" #,cuda__root=${cudatoolkit}
                    #"--set CUDA_HOME ${cudatoolkit}"
                    #"--set CUDA_INC_DIR ${cudatoolkit}/include"
                  ];
                });

                # My guess : poetry2nix makes one of duplicated packges to null 
                # Here, jupyter_core, jupyter-core are duplicated so make jupyter_core null
                # Solution : bring jupyter_core back like below 
                #jupyter_core = pyprev.jupyter-core;
                # Nope, below is better
                #jupyter_core = (pyfinal.python_selected.jupyter_core.override {
                jupyter_core = (pkgs_2211.python39Packages.jupyter_core.override {
                # 5.0.0 : nixos-22.11<
                  inherit (pyfinal)
                    hatchling
                    traitlets
                    pytestCheckHook
                  ;
                }).overridePythonAttrs (old: rec {
                  #version = pyprev.jupyter-core.version;
                  #src = pyprev.jupyter-core.src;
                  #version = pyprev.jupyter_core.version;
                  #src = pyprev.jupyter_core.src;
                  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.nativeBuildInputs or [ ]) ++ [ pyfinal.platformdirs ]);
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.platformdirs ]);
                });

                jupyter-core = pyfinal.jupyter_core;

                matplotlib = (pyfinal.python_selected.matplotlib.override {
                #matplotlib = (pkgs_2211.python39Packages.matplotlib.override {
                # 3.6.2 : nixos-22.11<
                  inherit (pyfinal)
                    pycairo
                    cycler
                    python-dateutil
                    numpy
                    pyparsing
                    sphinx
                    tornado
                    kiwisolver
                    mock
                    pytz
                    pygobject3
                    certifi
                    pillow
                    fonttools
                    setuptools-scm
                    setuptools-scm-git-archive
                    packaging
                    tkinter
                    pyqt5
                  ;
                }).overridePythonAttrs (old: rec {
                  version = pyprev.matplotlib.version;
                  src = pyprev.matplotlib.src;
                  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [
                    final.ghostscript
                  ]);
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.contourpy ]);
                });

                #  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [ final. ]);
                #  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.nativeBuildInputs or [ ]) ++ [ final. ]);
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal. ]);
                #});

                # = (pyfinal.python_selected..override {
                #meson-python = (pkgs_2211.python39Packages.meson-python.override {
                ### 1.23.0 : nixos-22.05< <22.11<
                ##  inherit (pyfinal)
                ##  ;
                #}).overridePythonAttrs (old: rec {
                #  version = pyprev.meson-python.version;
                #  src = pyprev.meson-python.src;
                #});

                #nbclient = (pyfinal.python_selected.nbclient.override {
                nbclient = (pkgs_2211.python39Packages.nbclient.override {
                # 0.7.0 : nixos-22.11<
                  inherit (pyfinal)
                    async_generator
                    ipykernel
                    ipywidgets
                    jupyter-client
                    nbconvert
                    nbformat
                    nest-asyncio
                    pytest-asyncio
                    pytestCheckHook
                    traitlets
                    xmltodict
                  ;
                }).overridePythonAttrs (old: rec {
                  version = pyprev.nbclient.version;
                  src = pyprev.nbclient.src;
                });

                #nbdev = pyprev.nbdev.overridePythonAttrs (old: rec {
                #  # buildInputs, nativeBuildInputs, propagatedNativeBuildInputs doesn't work.
                #  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [
                #    self.packages.x86_64-linux.quarto
                #  ]);
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [
                #    pyfinal.twine
                #  ]);
                #});

                #nbformat = (pyfinal.python_selected.nbformat.override {
                nbformat = (pkgs_2211.python39Packages.nbformat.override {
                # 5.7.0 : nixos-22.11
                  inherit (pyfinal)
                    hatchling
                    hatch-nodejs-version
                    fastjsonschema
                    jsonschema
                    jupyter_core
                    traitlets
                    pep440
                    testpath
                  ;
                  pytestCheckHook = null;
                  #hatch-nodejs-version = pkgs_2211.python39Packages.hatch-nodejs-version;
                }).overridePythonAttrs (old: rec {
                  version = pyprev.nbformat.version;
                  src = pyprev.nbformat.src;
                });

                ##numba = pkgs_2211.python39Packages.numba.override {
                #numba = pyfinal.python_selected.numba.override {
                #  inherit (pyfinal)
                #    numpy
                #    setuptools
                #    llvmlite;
                #    #importlib-metadata;
                #  python = pyfinal.python_selected;
                #  cudaPackages = final.cudaPackages;
                #  cudaSupport = true;
                #};

                #numpy = (pkgs_2211.python39Packages.numpy.override {
                # glibc_2.35 not found 
                numpy = (pyfinal.python_selected.numpy.override {
                # 1.23.0 : nixos-22.05< <22.11
                  inherit (pyfinal)
                    python
                    hypothesis
                    pytest
                    #typing-extensions
                    cython
                  ;
                  blas = final.blas_custom; # not prev.blas
                  lapack = final.lapack_custom; # not prev.blas
                }).overridePythonAttrs (old: rec {
                  version = pyprev.numpy.version;
                  src = pyprev.numpy.src;
                });

                # poetry.lock : 7.0.0 , nix repl : 8.0.0 because apache-beam (2.40.0) depends on pyarrow (>=0.15.1,<8.0.0)
                #pyarrow = pyfinal.python_selected.pyarrow.override {
                #  inherit (pyfinal)
                #    cffi
                #    cloudpickle
                #    cython
                #    fsspec
                #    hypothesis
                #    numpy
                #    pandas
                #    pytestCheckHook
                #    pytest-lazy-fixture
                #    scipy
                #    setuptools-scm
                #    six;
                #  python = pyfinal.python_selected;
                #};

                # = (pyfinal.python_selected..override {
                # = (pkgs_2211.python39Packages..override {
                ## 1.23.0 : nixos-22.05< <22.11<
                #  inherit (pyfinal)
                #  ;
                #}).overridePythonAttrs (old: rec {
                #  version = pyprev..version;
                #  src = pyprev..src;
                #});

                # = (pyfinal.python_selected..override {
                poetry = (pkgs_2211.python39Packages.poetry.override {
                # 1.2.2 : nixos-22.05< <22.11<
                  #inherit (pyfinal)
                  #;
                }).overridePythonAttrs (old: rec {
                #  version = pyprev..version;
                #  src = pyprev..src;
                });

                PyTDC = pyprev.pytdc;
                
                pytorch = pyfinal.torch;

                #scipy = pyprev.scipy.overridePythonAttrs (old: rec {
                #  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [ final. ]);
                #  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.nativeBuildInputs or [ ]) ++ [ pkgs_2211.python39Packages.meson-python final.pkg-config ]);
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal. ]);
                #});
                #scipy = (pkgs_2211.python39Packages.scipy.override {
                scipy = (pyfinal.python_selected.scipy.override {
                # 1.9.3 : nixos-22.11< # not working
                # 1.9.1 : nixos-22.11 # not working
                # 1.8.0 : nixos-22.05
                  inherit (pyfinal)
                    python
                    cython
                    pythran
                    #meson-python
                    nose
                    pytest
                    pytest-xdist
                    numpy
                    pybind11
                  ;
                  #meson-python = pkgs_2211.python39Packages.meson-python;
                }).overridePythonAttrs (old: rec {
                  #version = pyprev.scipy.version; # format issue
                  #src = pyprev.scipy.src;
                });

                seaborn = pyprev.seaborn.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.flit-core ]);
                });

                #tensorboard : tensorflow in pyproproject

                #tensorflow-gpu = (pkgs_2211.python39Packages.tensorflow.override {
                tensorflow-gpu = (pyfinal.python_selected.tensorflowWithCuda.override {
                # 2.10.0 : nixos-22.11
                # failed :Compiling tensorflow/lite/delegates/telemetry.cc failed: (Exit 1): crosstool_wrapper_driver_is_not_gcc failed:
                # 2.8.1 : nixos-22.05
                #bazel_5, buildBazelPackage, isPy3k, lib, fetchFromGitHub, symlinkJoin
                #which binutils perl
                ## Common libraries
                #jemalloc mpi grpc sqlite boringssl jsoncpp nsync
                #curl snappy lmdb-core icu double-conversion libpng libjpeg_turbo giflib;

                  inherit (pyfinal)
                    python
                    pybind11 cython
                    numpy tensorboard absl-py
                    #packaging
                    setuptools wheel keras keras-preprocessing google-pasta
                    opt-einsum astunparse h5py
                    termcolor grpcio six wrapt tensorflow-estimator
                    dill portpicker tblib typing-extensions
                    gast;
                  #glibcLocales = pkgs_2211.glibcLocales;
                  protobuf-python = pyfinal.protobuf;
                  protobuf-core = final.protobuf;
                  flatbuffers-python = pyfinal.flatbuffers;
                  flatbuffers-core = final.flatbuffers;
                  mklSupport = true;
                  mkl = final.mkl;
                }).overridePythonAttrs (old: {
                  #nativeBuildInputs = builtins.filter (x: ! builtins.elem (x.name or x.pname or "") [ "hook" ]) ((old.nativeBuildInputs or [ ]) ++ [
                    #pyfinal.setuptools # for pkgs_2211?
                  #]);
                  #dontUsePipInstall = true; # doesn't work
                  #version = pyprev.tensorflow-gpu.version; # can't read setup.py
                  #src = pyprev.tensorflow-gpu.src;
                });

                tensorflow-datasets = pyprev.tensorflow-datasets.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [  ]) ++ [
                    pyfinal.tensorflow-gpu
                  ]);
                });

                tensorflow-io-gcs-filesystem = pyprev.tensorflow-io-gcs-filesystem.overridePythonAttrs (old: {                                                                                                                     propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal.numpy ]);                                                                                    buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [ final.libtensorflow ]);                                                                                                });

                #tensorflow-metadata = (pyfinal.python_selected.tensorflow-metadata.overridePythonAttrs (old: rec {
                tensorflow-metadata = pyprev.tensorflow-metadata.overridePythonAttrs (old: rec {
                  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.nativeBuildInputs or [ ]) ++ [ pyfinal.pythonRelaxDepsHook ]);
                  pythonRelaxDeps = [ "protobuf" ];
                });

                termcolor = pyprev.termcolor.overridePythonAttrs (old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [
                    pyfinal.hatchling
                    # 'hatchling.build' has no attribute 'prepare_metadata_for_build_wheel': needs hatch-vcs not importlib-metadata
                    pyfinal.hatch-vcs
                  ]);
                });

                torch = (pyprev.torch.override {
                  inherit cudatoolkit; 
                  #cudatoolkit = final.cudaPackages_11_6.cudatoolkit; 
                  enableCuda = true;
                }).overrideAttrs (old: rec {
                  buildInputs = [
                    pyfinal.setuptools-scm
                    pyfinal.typing-extensions
                    final.linuxPackages.nvidia_x11
                    final.cudaPackages_11_6.nccl.dev
                    final.cudaPackages_11_6.nccl.out
                  ];
                #  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.nativeBuildInputs or [ ]) ++ [ final. ]);
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal. ]);
                });

                # nixos-22.11 : CHANGES NAME TO pyfinal.python_selected.torch!
                #torch = (pyfinal.python_selected.pytorch.override {
                #torch = (pkgs_2211.python39Packages.torch.override {
                ## 1.12.1 : nixos-22.11
                #  inherit (final)
                #    magma
                #    mpi
                #    cudaPackages
                #  ;
                #  inherit (pyfinal)
                #    # Native build inputs
                #    pybind11
                #    # Propagated build inputs
                #    numpy
                #    pyyaml
                #    cffi
                #    click
                #    typing-extensions
                #    # Unit tests
                #    hypothesis
                #    psutil
                #    # dependencies for torch.utils.tensorboard
                #    pillow
                #    six
                #    future
                #    tensorboard
                #    protobuf
                #  ;
                #  blas = final.blas_custom;
                #  MPISupport = true;
                #}).overridePythonAttrs (old: rec {
                #  # needed for 22.05 https://github.com/NixOS/nixpkgs/pull/175962
                #  USE_SYSTEM_BIND11 = true;
                #  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [
                #    #pyfinal.pybind11
                #    pyfinal.typing-extensions
                #  ]);
                #  #version = pyprev.torch.version;
                #  #src = pyprev.torch.src;
                #});

                #cu113 not working : changed to cu116
                #torch = (pyfinal.python_selected.pytorch-bin.override {
                #torch = (pkgs_2211.python39Packages.torch-bin.override {
                ## 1.12.1 : nixos-22.11
                #  inherit (pyfinal)
                #    python
                #    future
                #    numpy
                #    pyyaml
                #    requests
                #    setuptools
                #    typing-extensions
                #  ;
                #}).overridePythonAttrs (old: rec {
                ##  # needed for 22.05 https://github.com/NixOS/nixpkgs/pull/175962
                ##  USE_SYSTEM_BIND11 = true;
                ##  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [
                ##    pyfinal.pybind11
                ##  ]);
                #  version = pyprev.torch.version;
                #  src = pyprev.torch.src;
                #})

                #torch = nixpkgs.lib.makeOverridable
                #  ({ enableCuda ? true
                #   , cudatoolkit ? cudatoolkit
                #   , pkg ? pyprev.torch
                #   }: pkg.overrideAttrs (old:
                #    {
                #      preConfigure =
                #        if (!enableCuda) then ''
                #          export USE_CUDA=0
                #        '' else ''
                #          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${cudatoolkit}/targets/x86_64-linux/lib"
                #        '';
                #      preFixup = lib.optionalString (!enableCuda) ''
                #        # For some reason pytorch retains a reference to libcuda even if it
                #        # is explicitly disabled with USE_CUDA=0.
                #        find $out -name "*.so" -exec ${pkgs.patchelf}/bin/patchelf --remove-needed libcuda.so.1 {} \;
                #      '';
                #      buildInputs =
                #        (old.buildInputs or [ ])
                #        ++ [ self.typing-extensions ]
                #        ++ lib.optionals enableCuda [
                #          pkgs.linuxPackages.nvidia_x11
                #          pkgs.nccl.dev
                #          pkgs.nccl.out
                #        ];
                #      propagatedBuildInputs = [
                #        self.numpy
                #        self.future
                #        self.typing-extensions
                #      ];
                #    })
                #  )
                #  { };

                
                #torch-scatter = pyprev.torch-scatter.overridePythonAttrs (old: rec {
                #  BUILD_NAMEDTENSOR = "1";
                #  USE_MKLDNN = "1";
                #  USE_MKLDNN_CBLAS = "1";
                #  USE_SYSTEM_BIND11 = "1";
                #  USE_SYSTEM_NCCL = "1";
                #  uildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [ cudnn final.magma nccl final.numactl ]);
                #  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.nativeBuildInputs or [ ]) ++ [ cudatoolkit_joined ]);
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ final.mpi ]);
                #});

                quarto = pyprev.quarto.override {
                  preferWheel = true;
                };

                xarray-einstats = pyprev.xarray-einstats.overridePythonAttrs (old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ pyfinal.flit-core ];
                });

                # = (pyfinal.python_selected..override {
                xgboost = (pkgs_2211.python39Packages.xgboost.override {
                # 1.23.0 : nixos-22.05< <22.11
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
                }).overridePythonAttrs (old: rec {
                  version = pyprev.xgboost.version;
                  src = pyprev.xgboost.src;
                });

              }));
          });
        })
      ] ++ (builtins.attrValues jupyterWith.overlays)
        #++ [
        #(final: prev: {
        #  jupyterWith_python_custom = final.jupyterWith.override {
        #    python3 = self.python_custom.x86_64-linux;
        #  };
        #})
        #]
      );
    } // (flake-utils.lib.eachDefaultSystem (system: 
      rec
      {
        #jupyterWith_custom = jupyterWith.;

        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
          config.mklSupport = true;
          overlays = [ self.overlay ];
        };

        python_custom = pkgs.poetry2nix.mkPoetryEnv rec {
          projectDir = ./.;
          python = pkgs.python39;
          editablePackageSources = {
            mypackages = "~/Jupyter_Python"; #./.; not working
          };
        };

        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
        depNames = builtins.attrNames pyproject.tool.poetry.dependencies;

        iPythonWithPackages = pkgs.jupyterWith.kernels.iPythonWith {
          name = "my-awesome-python-env";
          python3 = python_custom;
          packages = p:
          let
              ## Building the local package using the standard way.
              #myPythonPackage = p.buildPythonPackage {
              #  pname = "MyPythonPackage";
              #  version = "1.0";
              #  src = ./my-python-package;
              #};
              poetryDeps =
                builtins.map (name: builtins.getAttr name p) depNames;
          in
          poetryDeps; #++ [ myPythonPackage ];
        };

        jupyterEnvironment = (pkgs.jupyterWith.jupyterlabWith {
          kernels = [ iPythonWithPackages ];
          #extraPackages = ps: [
          #];
        });

        apps.jupyter-lab = {
          type = "app";
          program = "${jupyterEnvironment}/bin/jupyter-lab";
        };

        packages.default = packages.poetry;
        packages.python_shell = pkgs.python39;
        packages.poetry = nixpkgs.legacyPackages.${system}.poetry;
        #packages.jupyterlab = jupyterEnvironment;
        packages.nbdev = python_custom.pkgs.nbdev;

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

        devShells.default = pkgs.mkShell rec {
          packages = [
            python_custom
            jupyterEnvironment
            pkgs.ffmpeg
          ];

          shellHook = ''export LD_LIBRARY_PATH='' + LD_LIBRARY_PATH + ''
            export LD_LIBRARY_PATH=${nixpkgs.lib.makeLibraryPath [ pkgs.cudaPackages.cudatoolkit "${pkgs.cudaPackages.cudatoolkit}" pkgs.cudaPackages.cudnn pkgs.nvidia_custom ]}:$LD_LIBRARY_PATH
            export TF_ENABLE_ONEDNN_OPTS=0 # when using GPU, oneDNN off is recommended 
            export PYTHONPATH=~/Jupyter_Python/nbdev_cards:$PYTHONPATH
          '';
        };
      }
    ));
}
####################################################################################################
# How poetry2nix works
#
#   nixpkgs    flake.nix   poetry.lock
#      O           X           X        Use current pkgs' python packages
#      O           X           O        Automatically make packages by inferring dependencies from poetry.lock
#      O           O           X        Use the overriden code from flake.nix
#      O           O           O        Use the overriden code from flake.nix
#      X           X           O        ? only example : etils -> infinite recursion
#      X           O           X        Use the overriden code from flake.nix
#      X           O           Q        Use the overriden code from flake.nix
#
# - If certain package is not in poetry.lock, that package is not importable!
# 
# Workflow
#
# 1. Packages that needs special care (ex. cuda related datascience packages) : 
#    pin them to existing nixos-unstable/nixos-YY.MM in the flake.nix
#
# 2. Try packaging only with poetry.lock as much as possible
#    - no module found : try solving with pyprev
#
# 3. Try packaging with nixos-unstable with version, src unlimited
#    - pkgs_2211.python39Packages..override {
#      }).overridePythonAttrs (old: rec {
#        version = pyprev..version;
#        src = pyprev..src;
#      });
#
# 4. Try packaging with nixos-unstable with version, src fixed to nixpkgs-unstable
#    - pkgs_2211.python39Packages..override {
#      }
#
# 5. Try packaging with nixos-YY.MM with version, src unlimited
#    - pyfinal.python_selected..override {
#      }).overridePythonAttrs (old: rec {
#        version = pyprev..version;
#        src = pyprev..src;
#      });
#
# 6. Try packaging with nixos-YY.MM with version, src fixed to nixos-YY.MM
#    - pyfinal.python_selected..override {
#      }
#
####################################################################################################
#
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
#   export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib64:${pkgs.cudaPackages.cudatoolkit.lib}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.nvidia_custom}/lib:${nixpkgs.lib.makeLibraryPath [ pkgs.cudaPackages.cudatoolkit "${pkgs.cudaPackages.cudatoolkit}" ]}:$LD_LIBRARY_PATH
#   - new version :
#   export LD_LIBRARY_PATH=${nixpkgs.lib.makeLibraryPath [ pkgs.cudaPackages.cudatoolkit "${pkgs.cudaPackages.cudatoolkit}" pkgs.cudaPackages.cudnn pkgs.nvidia_custom ]}:$LD_LIBRARY_PATH
# ${nixpkgs.lib.makeLibraryPath [ pkgs.cudaPackages.cudatoolkit ]} == ${pkgs.cudaPackages.cudatoolkit.lib}/lib
# ${nixpkgs.lib.makeLibraryPath [ "${pkgs.cudaPackages.cudatoolkit}" ]} == ${pkgs.cudaPackages.cudatoolkit}/lib
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
#   xgboost = pyfinal.python_selected.xgboost;
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
# - Use `pyfinal.python_selected.XXXX.override`
# - Examples : setuptools, setuptools-scm, pip
#
# Error: No module named 'pkg_resources' when running nbdev inside devShell
#
# Solution :
# - We need both `python_custom` and `python_custom.pkgs.nbdev` listed in devShell
#
####################################################################################################
                #
                # = pyprev..overridePythonAttrs (old: rec {
                #  buildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.buildInputs or [ ]) ++ [ final. ]);
                #  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.nativeBuildInputs or [ ]) ++ [ final. ]);
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyfinal. ]);
                #});
                # = (pyfinal.python_selected..override {
                # = (pkgs_2211.python39Packages..override {
                ## 1.23.0 : nixos-22.05< <22.11<
                #  inherit (pyfinal)
                #  ;
                #}).overridePythonAttrs (old: rec {
                #  version = pyprev..version;
                #  src = pyprev..src;
                #});
