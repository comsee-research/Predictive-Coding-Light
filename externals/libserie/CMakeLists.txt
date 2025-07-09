set(NEUVISYS_LIBSERIE_SOURCES
    comserie.cpp
    Faulhaber.cpp
)
    
set(NEUVISYS_LIBSERIE_HEADERS
    comserie.hpp
    Faulhaber.hpp
)
    
add_library(libserie ${NEUVISYS_LIBSERIE_SOURCES} ${NEUVISYS_LIBSERIE_HEADERS})


