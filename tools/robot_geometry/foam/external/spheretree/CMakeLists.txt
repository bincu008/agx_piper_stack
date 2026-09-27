set(SPHERETREE_SRC_FILES
     src/API/DomainSampler.cpp
     src/API/Hybrid.cpp
     src/API/MergeHubbard.cpp
     src/API/MSBase.cpp
     src/API/MSGrid.cpp
     src/API/MSRelaxation.cpp
     src/API/REBase.cpp
     src/API/REDiscard.cpp
     src/API/RELargest.cpp
     src/API/RELargestLM.cpp
     src/API/REMaxElim.cpp
     src/API/RESelect.cpp
     src/API/SEConvex.cpp
     src/API/SEPoint.cpp
     src/API/SESphPt.cpp
     src/API/SFBase.cpp
     src/API/SFMinErr.cpp
     src/API/SFRitter.cpp
     src/API/SFWhite/ball.cpp
     src/API/SFWhite/complex.c
     src/API/SFWhite.cpp
     src/API/SFWhite/nrutil.c
     src/API/SFWhite/svd.cpp
     src/API/SOBalance.cpp
     src/API/SOBase.cpp
     src/API/SOPerSphere.cpp
     src/API/SOSimplex.cpp
     src/API/SRBurst.cpp
     src/API/SRComposite.cpp
     src/API/SRExpand.cpp
     src/API/SRGrid.cpp
     src/API/SRMerge.cpp
     src/API/SRQuickBurst.cpp
     src/API/SRSpawn.cpp
     src/API/SRStandOff.cpp
     src/API/SRVoronoi.cpp
     src/API/SSIsohedron.cpp
     src/API/SSRings.cpp
     src/API/STGGeneric.cpp
     src/API/STGHubbard.cpp
     src/API/STGOctree.cpp
     src/API/SurfaceDiv.cpp
     src/API/VFAdaptive.cpp
     src/API/VFGapCross.cpp
     src/Base/ProcMon.cpp
     src/BBox/BBox.cpp
     src/BBox/eigen.cpp
     src/BBox/MVBB/gdiam.cpp
     src/BBox/qHull/geom2.cpp
     src/BBox/qHull/geom.cpp
     src/BBox/qHull/global.cpp
     src/BBox/qHull/mem.cpp
     src/BBox/qHull/merge.cpp
     src/BBox/qHull/poly2.cpp
     src/BBox/qHull/poly.cpp
     src/BBox/qHull/qhull.cpp
     src/BBox/qHull/set.cpp
     src/BBox/qHull/stat.cpp
     src/BBox/qHull/user.cpp
     src/DecodeParam.cpp
     src/EvalTree.cpp
     src/Exceptions/ErrorLog.cpp
     src/Exceptions/Exception.cpp
     src/Exceptions/MemoryException.cpp
     src/Export/POV.cpp
     src/Export/VRML.cpp
     src/Geometry/Circle.cpp
     src/Geometry/CubeTri.cpp
     src/Geometry/Line2D.cpp
     src/Geometry/LineSeg.cpp
     src/Geometry/pcube/fpcube.cpp
     src/Geometry/pcube/pcube.cpp
     src/Geometry/Plane.cpp
     src/Geometry/Point2D.cpp
     src/Geometry/Point3D.cpp
     src/Geometry/RayBox.c
     src/Geometry/Ray.cpp
     src/Geometry/Sphere.cpp
     src/Geometry/Transform2D.cpp
     src/Geometry/Transform3D.cpp
     src/Geometry/TriTri.cpp
     src/Geometry/Vector2D.cpp
     src/Geometry/Vector3D.cpp
     src/LinearAlgebra/Matrix.cpp
     src/MedialAxis/Clusters.cpp
     src/MedialAxis/MedialTester.cpp
     src/MedialAxis/Voronoi3D.cpp
     src/MedialAxis/VoronoiFace.cpp
     src/MinMax/Simplex.cpp
     src/null.c
     src/SphereTree/SphereTree.cpp
     src/Surface/Internal.cpp
     src/Surface/InternalGem.cpp
     src/Surface/OBJLoader.cpp
     src/Surface/SpacialHash.cpp
     src/Surface/Surface.cpp
     src/Surface/SurfaceRep.cpp
     src/Surface/SurfaceTester.cpp
     src/Surface/VolInt.cpp
     src/Surface/Voxel.cpp
     src/VerifyModel.cpp
 )

add_library(spheretree STATIC EXCLUDE_FROM_ALL ${SPHERETREE_SRC_FILES})

add_executable(makeTreeGrid src/makeTreeGrid.cpp)
add_executable(makeTreeHubbard src/makeTreeHubbard.cpp)
add_executable(makeTreeMedial src/makeTreeMedial.cpp)
add_executable(makeTreeOctree src/makeTreeOctree.cpp)
add_executable(makeTreeSpawn src/makeTreeSpawn.cpp)

target_link_libraries(makeTreeGrid PUBLIC spheretree)
target_link_libraries(makeTreeHubbard PUBLIC spheretree)
target_link_libraries(makeTreeMedial PUBLIC spheretree)
target_link_libraries(makeTreeOctree PUBLIC spheretree)
target_link_libraries(makeTreeSpawn PUBLIC spheretree)
