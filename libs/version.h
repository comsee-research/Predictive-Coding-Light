//! version of the neuvisys library
#define NEUVISYS_VERSION_MAJOR 1
#define NEUVISYS_VERSION_MINOR 0
#define NEUVISYS_VERSION_PATCH 0

//! version of the neuvisys library as string
#define NEUVISYS_VERSION "1.0.0"

//! version of the neuvisys library as an int, i.e., 100 * (100 * version_major() + version_minor()) + version_patch();
#define NEUVISYS_VERSION_INT (100 * (100 * NEUVISYS_VERSION_MAJOR + NEUVISYS_VERSION_MINOR) + NEUVISYS_VERSION_PATCH)
