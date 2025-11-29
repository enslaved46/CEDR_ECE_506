ARCH ?= x86

LIBDASH_INC ?= ../../libdash
LIBDASH_LIB ?= ../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

ifeq ($(ARCH), aarch64)
	CXX = aarch64-linux-gnu-g++
	CC = aarch64-linux-gnu-gcc
	RUNTIME_LIBDIR = ../../build-arm/libdash
	LIBS = -L $(RUNTIME_LIBDIR)
endif

ifeq ($(ARCH), arm32)
    CXX = /usr/bin/arm-linux-gnueabihf-g++
    CC = /usr/bin/arm-linux-gnueabihf-gcc
    RUNTIME_LIBDIR = ../../build-arm32/libdash
	LIBS = -L $(RUNTIME_LIBDIR)
endif

APP_NAME = read_temp
SOURCES  = read_temp_api.c
#SOURCES_NB = pulse_doppler_nb.c
#SOURCES_NB_STATIC_INPUT = pulse_doppler_nb_static_input.c

non-api: $(APP_NAME)_non_api.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) $(APP_NAME)_non_api.c -lm -lgsl -lgslcblas -o $(APP_NAME)_non_api-$(ARCH).out

api: $(APP_NAME)_api.c
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(APP_NAME)_api.c -o $(APP_NAME)_api-$(ARCH).so -O3
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) $(APP_NAME)_api.c -l:libdash.a -lm -o $(APP_NAME)_api-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
