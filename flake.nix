{
  description = "MS thesis environment";
  inputs = {
    #nixpkgs_edge.url = "github:sepiabrown/nixpkgs/poetry_edge_test2";#test_mkl_on_server_echo1";#NixOS/nixpkgs"; # poetry doesn't work at nixos-20.09
    nixpkgs.url = "github:sepiabrown/nixpkgs/poetry_test";#test_mkl_on_server_echo1";#NixOS/nixpkgs"; # poetry doesn't work at nixos-20.09
    #nixpkgs_2009.url = "nixpkgs/nixos-20.09"; # poetry doesn't work at nixos-20.09
    #nixpkgs_1703.url = "nixpkgs/1849e695b00a54cda86cb75202240d949c10c7ce"; # poetry doesn't work at nixos-20.09
    jupyterWith = {
      url = "github:tweag/jupyterWith";#/environment_variables_test"; 
      #inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
      #inputs.nixpkgs.follows = "nixpkgs";
    };
    dpaetzel.url = "github:dpaetzel/overlays/master";
  };

outputs = inputs@{ self, nixpkgs, jupyterWith, flake-utils, dpaetzel, ... }:
  let
    pkgs = import nixpkgs {
      system = "x86_64-linux";
      config.allowUnfree = true;
      config.cudaSupport = true;
      overlays = [ self.overlay ];
      #overlays = (builtins.attrValues jupyterWith.overlays) ++ [ self.overlay ]; # [ (import ./haskell-overlay.nix) ];
    };

    python_custom = pkgs.poetry2nix.mkPoetryEnv rec {
      projectDir = ./.;
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
          #[  ] ++ # adds nixpkgs.url version  python pkgs.
          [ myPythonPackage ] ++ poetryDeps; ### ++ (poetryExtraDeps p);
    };
    jupyterEnvironment = (pkgs.jupyterlabWith {
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
  in 
    {
      overlay = nixpkgs.lib.composeManyExtensions ([
        (self: super: { 
          polynote = self.stdenv.mkDerivation rec {
            pname = "polynote";
            version = "0.4.5";
            src = builtins.fetchurl {
              url = "https://github.com/polynote/polynote/releases/download/0.4.5/polynote-dist.tar.gz";
              sha256 = "sha256:1m7braf2d9gmqz270xrj0w4bc3j58bz0mx3h1ik9p1221dz2xc1j";#super.lib.fakeSha256;
            };
            #buildInputs = ( with self.python3.pkgs; [ virtualenv ipython nbconvert numpy pandas jedi jep ]);
            buildInputs = ( with python_custom.pkgs; [ virtualenv ipython nbconvert numpy pandas jedi jep ]);
            #nativeBuildInputs = [ self.python3.pkgs.jep ];
            #echo -e "`ls -al`\n"
            #postUnpack = ''
            #  chmod +x polynote
            #'';
            patchPhase = ''
              echo -e "`find ${python_custom.pkgs.jep}`\n"
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              ldd ${python_custom.pkgs.jep}/lib/python3.9/site-packages/jep/libjep.so
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
              substituteInPlace polynote.py \
                --replace 'os.path.dirname(os.path.realpath(__file__))' 'os.getcwd()' \
                --replace 'p.joinpath("jep") for p in paths if p.joinpath("jep").exists()' '"${python_custom.pkgs.jep}/lib/python3.9/site-packages/jep"' \
                --replace 'cmd = f"java' 'cmd = f"${self.openjdk8}/bin/java' \
                --replace '-Djava.library.path={jep_path}' '-Djava.library.path=${self.openjdk8}:${self.glibc}/lib:{jep_path}'
            '';
            installPhase = ''
              mkdir -p $out/bin
              cp polynote.py $out/bin/polynote
              chmod +x $out/bin/polynote
            '';
            #makeWrapperArgs = [ "--set PYTHONPATH ~/.cache/pypoetry/virtualenvs/my-python-package-1TYYO27q-py3.9/lib/python3.9/site-packages"];
            shellHook = ''
              #patchelf --set-rpath "${self.glibc}/lib" "${python_custom.pkgs.jep}/lib/python3.9/site-packages/jep/libjep.so"
              #export PYTHONPATH=~/.cache/pypoetry/virtualenvs/my-python-package-1TYYO27q-py3.9/lib/python3.9/site-packages
            '';
          };

          nvidia_custom = super.linuxPackages.nvidia_x11.overrideAttrs (oldAttrs: rec {
            version = "495.29.05";
            src = builtins.fetchurl {
              url = "https://us.download.nvidia.com/XFree86/Linux-x86_64/${version}/NVIDIA-Linux-x86_64-${version}.run";
              sha256 = "sha256-9yVLl9QAxpJQR5ZJb059j2TpOx4xxCeGCk8hmhhvEl4=";#super.lib.fakeSha256;
            };
          }); 
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
          #cudaPackages = super.cudaPackages_11_5;
          #cudnn_custom = super.cudaPackages_11_6.cudnn.overrideAttrs (old: {
          #  src = builtins.fetchurl { 
          #    url = old.src.url;
          #    sha256 = "sha256-YAJn8squ0v1Y6yFLpmnY6jXzlqfRm5SCLms2+fcIjCA=";
          #  };
          #});
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
          #jep = self.python3.pkgs.buildPythonPackage rec {
          #  pname = "jep";
          #  version = "4.0.3";
          #  src = self.fetchFromGitHub {
          #    owner = "ninia";
          #    repo = pname;
          #    rev = "v${version}";
          #    sha256 = "sha256-HYWC8FOeyHCzajnO9pQToK7KHRZ0TMNVQo9uy9d2CaY=";
          #  };

          #  doCheck = false;

          #  checkInputs = [ python-self.numpy ];
          #  propagatedBuildInputs = [ python-self.setuptools python-self.numpy ];
          #  buildInputs = [ self.glibc self.openjdk8 ];
          #  postInstall = ''
          #    echo "start"
          #    ldd $out/lib/python3.9/site-packages/jep/libjep.so
          #    echo "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
          #    echo "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
          #    echo "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
          #    echo "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
          #    echo "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
          #    #patchelf --set-interpreter ${self.glibc}/lib/ld-linux-x86-64.so.2 "$out/lib/python3.9/site-packages/jep/libjep.so"
          #    echo "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
          #    echo "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
          #    echo "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
          #    echo "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
          #    ldd $out/lib/python3.9/site-packages/jep/libjep.so
          #    echo "done"
          #  '';
          #};
          #).overridePythonAttrs (old: {
          #  installCheckPhase = ''
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    ldd $out/lib/python3.9/site-packages/jep/libjep.so
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #    echo "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII'
          #  '';
          #});
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
                  { ${blasImplementation} = {
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
          jupyterlab = python-super.jupyterlab.overridePythonAttrs (oldAttrs: {
            makeWrapperArgs = (oldAttrs.makeWrapperArgs or []) ++ [
                "--set LD_LIBRARY_PATH ${self.nvidia_custom}/lib:$LD_LIBRARY_PATH" # ${self.cudaPackages_10_2.cudatoolkit}/targets/x86_64-linux/lib/stubs:${self.cudaPackages_11_6.cudatoolkit}/lib/stubs:
                "--set XLA_FLAGS --xla_gpu_cuda_data_dir=${self.cudaPackages_11_6.cudatoolkit}"
                #"--set AESARA_FLAGS device=cuda0,dnn__base_path=${self.cudaPackages_11_6.cudnn},blas__ldflags=-lblas,dnn__library_path=${self.cudaPackages_11_6.cudnn}/lib,dnn__include_path=${self.cudaPackages_11_6.cudnn}/include"#${nixpkgs.lib.makeLibraryPath [ self.cudaPackages.cudnn ]}" #,cuda__root=${self.cudaPackages_11_6.cudatoolkit}
                #"--set CUDA_HOME ${self.cudaPackages_11_6.cudatoolkit}"
                #"--set CUDA_INC_DIR ${self.cudaPackages_11_6.cudatoolkit}/include"
            ];
          });
        });
      });
          poetry2nix = super.poetry2nix.overrideScope' (p2nixself: p2nixsuper: {
          # pyself & pysuper refers to python packages
            defaultPoetryOverrides = p2nixsuper.defaultPoetryOverrides.extend (pyself: pysuper: 
              let
                torch_custom = nixpkgs.lib.makeOverridable (
                  { enableCuda ? true
                  , cudaPackages ? self.cudaPackages #, cudatoolkit ? pkgs.cudatoolkit_10_1
                  , cudaArchList ? null
                  , pkg ? pysuper.torch
                  }:
                  let
                    inherit (cudaPackages) cudatoolkit cudnn nccl cuda_nvcc;

                    cudatoolkit_joined = self.symlinkJoin {
                      name = "${cudatoolkit.name}-unsplit";
                      # nccl is here purely for semantic grouping it could be moved to nativeBuildInputs
                      paths = [ cudatoolkit.out cudatoolkit.lib nccl.dev nccl.out ];
                    };

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
                      if !enableCuda || cudaArchList != null
                      then cudaArchList
                      else cudaCapabilities."cuda${nixpkgs.lib.versions.major cudatoolkit.version}";
                  in
                  pkg.overrideAttrs (old: {
                    src_test = old.src;

                    preConfigure = nixpkgs.lib.optionalString (!enableCuda) ''
                      export USE_CUDA=0
                    '' + nixpkgs.lib.optionalString enableCuda ''
                      export TORCH_CUDA_ARCH_LIST="${nixpkgs.lib.strings.concatStringsSep ";" final_cudaArchList}"
                      export CC=${cudatoolkit.cc}/bin/gcc CXX=${cudatoolkit.cc}/bin/g++
                      export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${nixpkgs.lib.makeLibraryPath [ cudatoolkit "${cudatoolkit}" ]}"
                    '' + nixpkgs.lib.optionalString (enableCuda && cudnn != null) ''
                      export CUDNN_INCLUDE_DIR=${cudnn}/include
                    ''; # enableCuda ${cudatoolkit}/targets/x86_64-linux/lib

                      # patchelf --set-rpath "${lib.makeLibraryPath [ stdenv.cc.cc.lib ]}"
                    preFixup = nixpkgs.lib.optionalString (!enableCuda) ''
                      # For some reason pytorch retains a reference to libcuda even if it
                      # is explicitly disabled with USE_CUDA=0.
                      find $out -name "*.so" -exec ${nixpkgs.patchelf}/bin/patchelf --remove-needed libcuda.so.1 {} \;
                    '';
                    nativeBuildInputs =
                      (old.nativeBuildInputs or [ ])
                      ++ [ self.autoPatchelfHook ]
                      ++ nixpkgs.lib.optionals enableCuda [ cudatoolkit_joined self.addOpenGLRunpath ];
                    buildInputs =
                      (old.buildInputs or [ ])
                      ++ [ pyself.typing-extensions pyself.pyyaml ]
                      ++ nixpkgs.lib.optionals enableCuda [
                        self.nvidia_custom
                        nccl.dev
                        nccl.out
                        cudatoolkit
                        cudnn
                        cuda_nvcc
                        self.magma
                        nccl
                      ];
                    propagatedBuildInputs = [
                      pyself.numpy
                      pyself.future
                      pyself.typing-extensions
                    ];
                  })
                ) 
                { };
              in
              {
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
                # gast 0.5.0 needed by beniget needed by pythran needed by scipy
                # but tensorflow needs gast <= 0.4.0
                beniget = pysuper.beniget.overridePythonAttrs ( old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ pysuper.gast ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyself.gast_5 ]);
                });
                gast_4 = pysuper.gast.overridePythonAttrs ( old: rec {
                  pname = "gast";
                  version = "0.4.0";

                  src = self.fetchFromGitHub {
                    owner = "serge-sans-paille";
                    repo = pname;
                    rev = version;
                    sha256 = "sha256-vmjx/cULyvM6q1ZzQnQS4VkeXSto8JHZzS8PGRFQDH4=";
                  };
                });
                gast_5 = pysuper.gast.overridePythonAttrs ( old: rec {
                  pname = "gast";
                  version = "0.5.0";

                  src = self.fetchFromGitHub {
                    owner = "serge-sans-paille";
                    repo = pname;
                    rev = version;
                    sha256 = "sha256-fGkr1TIn1NPlRh4hTs9Rh/d+teGklrz+vWKCbacZT2M=";
                  };
                });
                jax = (self.python3.pkgs.jax.override {
                  inherit (pyself)
                  absl-py
                  numpy
                  opt-einsum
                  scipy
                  typing-extensions
                  jaxlib;
                  blas = self.blas_custom;
                  lapack = self.lapack_custom;
                }).overridePythonAttrs ( old: {
                  propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [
                    pyself.jaxlib
                  ];
                });
                  
                jaxlib = (self.python3.pkgs.jaxlib.override {
                  inherit (pyself) 
                  absl-py
                  #double-conversion
                  flatbuffers
                  #giflib
                  #grpc
                  #jsoncpp
                  #libjpeg_turbo
                  numpy
                  scipy
                  six;
                  #snappy;
                  
                  cudaSupport = true;
                  cudaPackages = self.cudaPackages_11_6;
                  mklSupport = true;
                });
                #.overridePythonAttrs ( old: {
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x (with self.python3.pkgs; [ scipy numpy six ])) ((old.propagatedBuildInputs or [ ]) ++ [ ]);
                #});
                jep = pysuper.jep.overridePythonAttrs (old: {
                  propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ 
                    pyself.setuptools
                    pyself.numpy
                  ];
                  buildInputs = (old.buildInputs or [ ]) ++ [ 
                    self.glibc
                    self.openjdk8
                  ];
                });
                #libgpuarray = pysuper.libgpuarray.overridePythonAttrs ( old: {
                #  libraryPath = nixpkgs.lib.makeLibraryPath (with self.cudaPackages_11_6; [ cudnn cudatoolkit.lib cudatoolkit.out self.clblas self.ocl-icd ]);
                #  buildInputs = (old.buildInputs or [ ]) ++ [ self.cudaPackages_11_6.cudatoolkit self.cudaPackages_11_6.cudnn pyself.nose];
                #  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                #    self.cudaPackages_11_6.cudatoolkit self.cudaPackages_11_6.cudnn pyself.nose
                #  ];
                #});
                mkl-service = pysuper.mkl-service.overridePythonAttrs (old: {
                  patchPhase = ''
                      substituteInPlace mkl/_mkl_service.pyx \
                        --replace "'avx512_e4': mkl.MKL_ENABLE_AVX512_E4," " " \
                        --replace "'avx2_e1': mkl.MKL_ENABLE_AVX2_E1," " "

                      substituteInPlace mkl/_mkl_service.pxd \
                        --replace "int MKL_ENABLE_AVX512_E4" " " \
                        --replace "int MKL_ENABLE_AVX2_E1" " "
                  '';
                });
                pillow = pysuper.pillow.overridePythonAttrs ( old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ self.xorg.libxcb ];
                });
                poetry-core = pysuper.poetry-core.overridePythonAttrs (old: {
                  # 1.2.0b1
                  postPatch = nixpkgs.lib.optionalString (nixpkgs.lib.versionOlder self.python.version "3.8") ''
                    # remove >1.0.3
                    substituteInPlace pyproject.toml \
                      --replace 'importlib-metadata = {version = "^1.7.0", python = "~2.7 || >=3.5, <3.8"}' \
                        'importlib-metadata = {version = ">=1.7.0", python = "~2.7 || >=3.5, <3.8"}'
                  '';

                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
                    #pyself.intreehooks
                  ];

                  propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ nixpkgs.lib.optionals (nixpkgs.lib.versionOlder self.python.version "3.8") [
                    #`pyself.importlib-metadata
                  ] ++ nixpkgs.lib.optionals pyself.isPy27 [
                    pyself.pathlib2
                    pyself.typing
                  ];

                  checkInputs = [
                    nixpkgs.git
                    pyself.pep517
                    pyself.pytest-mock
                    pyself.pytestCheckHook
                    pyself.tomlkit
                    pyself.virtualenv
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
                #Theano = pysuper.Theano.overridePythonAttrs ( old: {
                #  propagatedBuildInputs = [ 
                #    pyself.libgpuarray
                #    pyself.numpy
                #    pyself.numpy.blas
                #    pyself.scipy
                #    pyself.setuptools
                #    pyself.six
                #    pyself.pandas
                #    pyself.filelock
                #    self.cudaPackages_11_6.cudatoolkit
                #    self.cudaPackages_11_6.cudnn
                #  ];
                #});

                # needed for pymc3
                theano-pymc = pysuper.theano-pymc.overridePythonAttrs ( old: 
                let
  wrapped = command: buildTop: buildInputs:
    self.runCommandCC "${command}-wrapped" { inherit buildInputs; } ''
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
    if self.stdenv.cc.isGNU then "g++" else
    if self.stdenv.cc.isClang then "clang++" else
    throw "Unknown C++ compiler";
  cxx_compiler = wrapped cxx_compiler_name "\\$HOME/.theano"
    ( [ pyself.libgpuarray self.cudaPackages_11_6.cudnn self.cudaPackages_11_6.cudatoolkit] );

                in
                {
  postPatch = ''
    substituteInPlace theano/configdefaults.py \
      --replace 'StrParam(param, is_valid=warn_cxx)' 'StrParam('\'''${cxx_compiler}'\''', is_valid=warn_cxx)' \
      --replace 'rc == 0 and config.cxx != ""' 'config.cxx != ""'
    substituteInPlace theano/configdefaults.py \
      --replace 'StrParam(get_cuda_root)' 'StrParam('\'''${self.cudaPackages_11_6.cudatoolkit}'\''')'
    substituteInPlace theano/configdefaults.py \
      --replace 'StrParam(default_dnn_base_path)' 'StrParam('\'''${self.cudaPackages_11_6.cudnn}'\''')'
  '';

  # needs to be postFixup so it runs before pythonImportsCheck even when
  # doCheck = false (meaning preCheck would be disabled)
  postFixup = ''
    mkdir -p check-phase
    export HOME=$(pwd)/check-phase
  '';
                    #preConfigure = 
                    #''
                    #  export CC=${self.cudaPackages.cudatoolkit.cc}/bin/gcc CXX=${self.cudaPackages.cudatoolkit.cc}/bin/g++
                    #  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${nixpkgs.lib.makeLibraryPath [ self.cudaPackages.cudatoolkit "${self.cudaPackages.cudatoolkit}" ]}"
                    #  export CUDNN_INCLUDE_DIR=${self.cudaPackages.cudnn}/include
                    #''; # enableCuda ${cudatoolkit}/targets/x86_64-linux/lib

                    #  # patchelf --set-rpath "${lib.makeLibraryPath [ stdenv.cc.cc.lib ]}"
                    #preFixup =  ''
                    #  # For some reason pytorch retains a reference to libcuda even if it
                    #  # is explicitly disabled with USE_CUDA=0.
                    #  find $out -name "*.so" -exec ${nixpkgs.patchelf}/bin/patchelf --remove-needed libcuda.so.1 {} \;
                    #'';
                  propagatedBuildInputs = [ 
                    pyself.libgpuarray
                    pyself.numpy
                    pyself.numpy.blas
                    pyself.scipy
                    pyself.setuptools
                    #pyself.six
                    pyself.pandas
                    pyself.filelock
                    self.cudaPackages_11_6.cudatoolkit
                    self.cudaPackages_11_6.cudnn
                  ];
                });
                traitlets = pysuper.traitlets.overridePythonAttrs ( old: {
                  propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [ pyself.hatchling ];
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
                # pythran needed by scipy
                pythran = pysuper.pythran.overridePythonAttrs ( old: rec {
                  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [ pysuper.gast ]) ((old.propagatedBuildInputs or [ ]) ++ [ pyself.gast_5 ]);
                    # needed for scipy 0.18.1
                    #pname = "pythran";
                    #version = "0.10.0";

                    #src = self.fetchFromGitHub {
                    #  owner = "serge-sans-paille";
                    #  repo = "pythran";
                    #  rev = version;
                    #  sha256 = "sha256-BLgMcK07iuRPBJqQpLXGnf79KgRsTqygCSeusLqkfxc=";
                    #};
                });
                setuptools = super.python3.pkgs.setuptools.overridePythonAttrs (old: {
                  catchConflicts = false;
                  format = "other";
                });
                scipy = pysuper.scipy.overridePythonAttrs ( old: {
                  doCheck = false;
                });
                #tensorflow = pysuper.tensorflow.overridePythonAttrs ( old: {
                #  nativeBuildInputs = builtins.filter (x: ! builtins.elem x [ pysuper.gast ]) ((old.nativeBuildInputs or [ ]) ++ [ pyself.wheel pyself.gast_4 ]);
                #});
                tensorflow-io-gcs-filesystem = pysuper.tensorflow-io-gcs-filesystem.overridePythonAttrs ( old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ self.libtensorflow ];
                });
                torch = torch_custom.override { enableCuda = true; };
                # jupyterWith also uses six causing six package collision in closure
                #six = self.python3.pkgs.six;
                xarray-einstats = pysuper.xarray-einstats.overridePythonAttrs ( old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ pyself.flit-core ];
                });
###############################################################
                #multipledispatch = pysuper.multipledispatch.overrideAttrs ( old: {
                #  #buildInputs = [ pysuper.six ];
                #  propagatedBuildInputs = builtins.filter (x: ! builtins.elem x [  ]) ((old.propagatedBuildInputs or [ ]) ++ [ self.python3.pkgs.six pysuper.six]);
                #});
                #logical-unification = pysuper.buildPythonPackage rec {
                #  pname = "logical-unification";
                #  version = "0.4.5";

                #  propagatedBuildInputs = [ pyself.multipledispatch pyself.toolz ];

                #  src = pysuper.fetchPypi {
                #    inherit pname version;
                #    sha256 = "sha256-fGpsG3xrqg9bmvk/Bs/I0kGba3kzRrZ47RNnwFznRVg=";
                #  };

                #  doCheck = false;
                #};

                #cons = pysuper.buildPythonPackage rec {
                #  pname = "cons";
                #  version = "0.4.5";

                #  propagatedBuildInputs = [ pyself.logical-unification ];

                #  src = pysuper.fetchPypi {
                #    inherit pname version;
                #    sha256 = "sha256-tGtIrbWlr39EN12jRtkm5VoyXU3BK5rdnyAoDTs3Qss=";
                #  };

                #  doCheck = false;
                #};

                #etuples = pysuper.buildPythonPackage rec {
                #  pname = "etuples";
                #  version = "0.3.4";

                #  propagatedBuildInputs = [ pyself.cons ];

                #  src = pysuper.fetchPypi {
                #    inherit pname version;
                #    sha256 = "sha256-mAUTeb0oTORi2GjkTDiaIiBrNcSVztJZBBrx8ypUoKM=";
                #  };

                #  doCheck = false;
                #};

                #miniKanren = pysuper.buildPythonPackage rec {
                #  pname = "miniKanren";
                #  version = "1.0.3";

                #  propagatedBuildInputs = [
                #    pyself.cons
                #    pyself.etuples
                #    pyself.logical-unification
                #  ];

                #  src = pysuper.fetchPypi {
                #    inherit pname version;
                #    sha256 = "sha256-Hsi9sBFErV6HUsfCl/uKEi25IPhZJ20lpy0WTpmNf24=";
                #  };

                #  doCheck = false;
                #};

                aesara =
                  let
  wrapped = command: buildTop: buildInputs:
    self.runCommandCC "${command}-wrapped" { inherit buildInputs; } ''
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
    if self.stdenv.cc.isGNU then "g++" else
    if self.stdenv.cc.isClang then "clang++" else
    throw "Unknown C++ compiler";
  cxx_compiler = wrapped cxx_compiler_name "\\$HOME/.theano"
    ( [ self.cudaPackages_11_6.cudnn self.cudaPackages_11_6.cudatoolkit] );

                in
                pysuper.buildPythonPackage rec {
  postPatch = ''
    substituteInPlace aesara/configdefaults.py \
      --replace 'StrParam(param, is_valid=warn_cxx)' 'StrParam('\'''${cxx_compiler}'\''', is_valid=warn_cxx)' \
      --replace 'rc == 0 and config.cxx != ""' 'config.cxx != ""'
    substituteInPlace aesara/configdefaults.py \
      --replace 'StrParam(get_cuda_root)' 'StrParam('\'''${self.cudaPackages_11_6.cudatoolkit}'\''')'
    substituteInPlace aesara/configdefaults.py \
      --replace 'StrParam(default_dnn_base_path)' 'StrParam('\'''${self.cudaPackages_11_6.cudnn}'\''')'
  '';

  # needs to be postFixup so it runs before pythonImportsCheck even when
  # doCheck = false (meaning preCheck would be disabled)
  postFixup = ''
    mkdir -p check-phase
    export HOME=$(pwd)/check-phase
  '';
                  pname = "aesara";
                  version = "2.4.0";

                  propagatedBuildInputs = [
                    pyself.minikanren#miniKanren
                    pyself.cons

                    pyself.numpy
                    pyself.numpy.blas
                    pyself.scipy
                    pyself.filelock

                    #pyself.libgpuarray
                    pyself.setuptools
                    #pyself.six
                    pyself.pandas
                    pyself.jaxlib
                    self.cudaPackages_11_6.cudatoolkit
                    self.cudaPackages_11_6.cudnn
                  ];

                  src = pysuper.fetchPypi {
                    inherit pname version;
                    sha256 = "sha256-DkiJtJjw1nQC+6k5P/Gk5WruwsB8RfWu6fB6iPvJzWQ=";
                  };

                  doCheck = false;
                };

                aeppl = pysuper.buildPythonPackage rec {
                  pname = "aeppl";
                  version = "0.0.26";

                  propagatedBuildInputs = [ pyself.aesara ];

                  src = pysuper.fetchPypi {
                    inherit pname version;
                    sha256 = "sha256-wuX0qqXThMW/ftOBCT/qIVdwB6EQnPGgo5XYcVY+D5w=";
                  };

                  doCheck = false;
                };

                pymc-nightly =
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
                  pysuper.buildPythonPackage rec {
                  pname = "pymc-nightly";
                  version = "4.0.0b2.dev20220304";

                  nativeBuildInputs = [ self.gfortran ];
                  buildInputs = [ blas lapack ];
                  enableParallelBuilding = true;
                  preBuild = ''
                    ln -s ${cfg} site.cfg
                  '';

                  meta.broken = false;

                  src = pysuper.fetchPypi {
                    inherit pname version;
                    sha256 = "sha256-U5s/HL8h7hLAOXWFlyvmbToqiZfEfRes3i57L+eCSJs=";
                  };

                  propagatedBuildInputs = [
                    pyself.arviz
                    pyself.cachetools
                    pyself.fastprogress
                    pyself.h5py
                    pyself.joblib
                    pyself.packaging
                    pyself.pandas
                    pyself.patsy
                    pyself.semver
                    pyself.tqdm
                    pyself.typing-extensions
                    pyself.jaxlib

                    pyself.cloudpickle

                    pyself.aeppl
                    pyself.aesara
                    pyself.numpy
                    blas
                    lapack
                    pyself.mkl-service
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

                  checkInputs = with pysuper; [ pytest pytest-cov ];
                };
            });
          });
        })
        
        #(final: pysuper: {
        #  # The application
        #  myapp = pysuper.poetry2nix.mkPoetryApplication {
        #    projectDir = ./.;
        #  };
        #})
        #jupyterWith.overlays.jupyterWith
        #jupyterWith.overlays.haskell
        #jupyterWith.overlays.python
      ] ++ (builtins.attrValues jupyterWith.overlays)
        #++ [ dpaetzel.pymc4 ] 
      );
      #overlay = nixpkgs.lib.composeManyExtensions ([
      #  (self: super: { 
      #    nvidia_custom = super.linuxPackages.nvidia_x11.overrideAttrs (oldAttrs: rec {
      #      version = "495.29.05";
      #      src = super.fetchurl {
      #        url = "https://us.download.nvidia.com/XFree86/Linux-x86_64/${version}/NVIDIA-Linux-x86_64-${version}.run";
      #        sha256 = "sha256-9yVLl9QAxpJQR5ZJb059j2TpOx4xxCeGCk8hmhhvEl4=";#super.lib.fakeSha256;
      #      };
      #    }); 
      #    blas_custom = (super.blas.override {
      #      blasProvider = self.mkl;
      #    }).overrideAttrs (oldAttrs: {
      #      buildInputs = (oldAttrs.buildInputs or [ ]) 
      #                    ++ self.lib.optional self.stdenv.hostPlatform.isDarwin self.fixDarwinDylibNames;
      #    });
      #    lapack_custom = (super.lapack.override {
      #      lapackProvider = self.mkl;
      #    }).overrideAttrs (oldAttrs: {
      #      buildInputs = (oldAttrs.buildInputs or [ ]) 
      #                    ++ self.lib.optional self.stdenv.hostPlatform.isDarwin self.fixDarwinDylibNames;
      #    });
      #    #cudaPackages = super.cudaPackages_11_5;
      #    python3 = super.python3.override (old: { # for jupyterWith!
      #      packageOverrides = 
      #        super.lib.composeExtensions
      #          (old.packageOverrides or (_: _: {}))
      #          (python-self: python-super: {
      #            httpx = python-super.httpx.overridePythonAttrs (old: { # for jupyterlab -> .. -> falcon
      #              doCheck = false;
      #            });
      #            httplib2 = python-super.httplib2.overridePythonAttrs ( old: {
      #              doCheck = false;
      #            });
      #            numpy = python-super.numpy.overridePythonAttrs ( old:
      #              let
      #                blas = self.blas_custom; # not super.blas
      #                lapack = self.lapack_custom; # not super.lapack
      #                blasImplementation = nixpkgs.lib.nameFromURL blas.name "-";
      #                cfg = super.writeTextFile {
      #                  name = "site.cfg";
      #                  text = (
      #                    nixpkgs.lib.generators.toINI
      #                      { }
      #                      {
      #                        ${blasImplementation} = {
      #                          include_dirs = "${blas}/include";
      #                          library_dirs = "${blas}/lib";
      #                        } // nixpkgs.lib.optionalAttrs (blasImplementation == "mkl") {
      #                          mkl_libs = "mkl_rt";
      #                          lapack_libs = "";
      #                        };
      #                      }
      #                  );
      #                };
      #              in
      #              {
      #                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ self.gfortran ];
      #                buildInputs = (old.buildInputs or [ ]) ++ [ blas lapack ];
      #                enableParallelBuilding = true;
      #                preBuild = ''
      #                  ln -s ${cfg} site.cfg
      #                '';
      #                passthru = old.passthru // {
      #                  blas = blas;
      #                  inherit blasImplementation cfg;
      #                };
      #              }
      #            );
      #            jupyterlab = python-super.jupyterlab.overridePythonAttrs (oldAttrs: {
      #              makeWrapperArgs = (oldAttrs.makeWrapperArgs or []) ++ [
      #                "--set LD_LIBRARY_PATH ${self.nvidia_custom}/lib:$LD_LIBRARY_PATH"
      #              ];
      #            });
      #          });
      #    });
      #  })
      #] ++  (builtins.attrValues jupyterWith.overlays));
    } // (flake-utils.lib.eachSystem [ "x86_64-linux" "x86_64-darwin" ] (system:
      rec 
      {
        inherit pkgs python_custom;
        apps.jupyterlab = {
          type = "app";
          program = "${jupyterEnvironment}/bin/jupyter-lab";
        };
        defaultApp = apps.jupyterlab;
        packages.poetry = pkgs.poetry;
        packages.mkl-service = python_custom.pkgs.mkl-service;
        packages.mkl = pkgs.mkl;
        packages.polynote = pkgs.polynote;
        packages.jep = pkgs.python3.pkgs.jep;
        defaultPackage = pkgs.poetry;
        #devShell = python_test.env.overrideAttrs (old: {
        #  nativeBuildInputs = with pkgs; old.nativeBuildInputs ++ [
        #    jupyterEnvironment
        #    poetry
        #    (lib.getBin caffe)
        #  ];
        #});
        packages.jupyterlab = pkgs.mkShell rec {
          packages = [ 
            pkgs.nvidia_custom
            jupyterEnvironment
            python_custom
            #iJulia.runtimePackages
          ];
          nativeBuildInputs = [ # hostPlatform, usually build time
          ];
          #JULIA_DEPOT_PATH = "./.julia_depot";
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.cudaPackages_11_6.cudatoolkit}/lib/stubs:${pkgs.nvidia_custom}/lib:$LD_LIBRARY_PATH
            export THEANO_FLAGS='device\=cuda0,dnn__base_path\=${pkgs.cudaPackages_11_6.cudnn},blas__ldflags\=-lblas,dnn__library_path\=${pkgs.cudaPackages_11_6.cudnn}/lib,dnn__include_path\=${pkgs.cudaPackages_11_6.cudnn}/include'
          '';
        };
        devShell = pkgs.mkShell rec {
          packages = [ 
            # stdenv reference :
            # https://discourse.nixos.org/t/nixos-with-poetry-installed-pandas-libstdc-so-6-cannot-open-shared-object-file/8442/3
            # https://nixos.wiki/wiki/Packaging/Quirks_and_Caveats#ImportError:_libstdc.2B.2B.so.6:_cannot_open_shared_object_file:_No_such_file
            #stdenv.cc.cc.lib
            
            jupyterEnvironment
            python_custom
            #(pkgs.lib.getBin pkgs.caffe)
            #(pkgs.lib.getBin python_custom)

            #iJulia.runtimePackages
          ];
          buildInputs = [ # hostPlatform, usually build time
            pkgs.polynote
          ];
          nativeBuildInputs = [ # hostPlatform, usually build time
          ];
          #JULIA_DEPOT_PATH = "./.julia_depot";

            #export PYTHONPATH="${python-with-my-packages}/${python-with-my-packages.sitePackages}:$PYTHONPATH"
            #export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
            #export PATH=${pkgs.cudatoolkit_11}/nsight-compute-2019.4.0:$(nixGLNvidia printenv PATH):${myNixGL}/bin
            #export LD_LIBRARY_PATH=$(nixGLNvidia printenv LD_LIBRARY_PATH)
            #export LD_LIBRARY_PATH=${pkgs.nvidia_x11}/lib:${pkgs.ncurses5}/lib
            #export LD_LIBRARY_PATH=${pkgs.libGL}/lib:${pkgs.libGLU}/lib:${pkgs.freeglut}/lib:${pkgs.xorg.libX11}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudatoolkit_11}/lib:${pkgs.cudatoolkit_11.lib}/lib:$LD_LIBRARY_PATH
            #export LD_LIBRARY_PATH=$(nixGLNvidia printenv LD_LIBRARY_PATH):$LD_LIBRARY_PATH
            #export CUDA_PATH=${pkgs.cudatoolkit_11}
            #export EXTRA_LDFLAGS="-L/lib -L${pkgs.nvidia_x11}/lib"
            #export EXTRA_CCFLAGS="-I/usr/include"
            #export LD_LIBRARY_PATH=${pkgs.libGL}/lib:${pkgs.libGLU}/lib:${pkgs.freeglut}/lib:${pkgs.xorg.libX11}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudatoolkit_11}/lib:${pkgs.cudatoolkit_11.lib}/lib:$LD_LIBRARY_PATH
            #export LD_LIBRARY_PATH=${pkgs.nvidia_custom}/lib:${pkgs.ncurses5}/lib
            #pkgs.linuxPackages.nvidia_x11
            #:${pkgs.ncurses5}/lib
            #LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/:$LD_LIBRARY_PATH
            #export CUDA_PATH=${pkgs.cudatoolkit_11_5}
            #export PATH=$CUDA_PATH:$PATH
            #export EXTRA_LDFLAGS="-L/lib -L${pkgs.nvidia_custom}/lib"
            #export EXTRA_CCFLAGS="-I/usr/include"
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.cudaPackages_11_6.cudatoolkit}/lib/stubs:${pkgs.nvidia_custom}/lib:$LD_LIBRARY_PATH
            #export THEANO_FLAGS='device=cuda0,dnn__base_path=${pkgs.cudaPackages_11_6.cudnn},blas__ldflags=-lblas,dnn__library_path=${pkgs.cudaPackages_11_6.cudnn}/lib,dnn__include_path=${pkgs.cudaPackages_11_6.cudnn}/include'
          '';
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
