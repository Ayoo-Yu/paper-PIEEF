#ifndef WIND_PHYSICS_ACTUATOR_MATH_H
#define WIND_PHYSICS_ACTUATOR_MATH_H

#include <cmath>

// Geometry is expressed in the same coordinates as the mesh. These ratios are
// fixed physical sampling/support definitions, independent of mesh.lc2.
namespace windPhysics
{
constexpr double diskHalfThicknessRatio = 0.25;
constexpr double referenceDistanceRatio = 1.0;
constexpr double referenceHalfThicknessRatio = 0.25;
constexpr double referenceRadiusRatio = 0.25;

inline bool insideAxialCylinder
(
    const double x, const double y, const double z,
    const double centreX, const double centreY, const double centreZ,
    const double halfThickness, const double radius
)
{
    const double dy = y - centreY;
    const double dz = z - centreZ;
    return std::fabs(x - centreX) <= halfThickness
        && dy*dy + dz*dz <= radius*radius;
}

inline double referenceMean(const double velocityVolumeSum, const double volume)
{
    return velocityVolumeSum / volume;
}

// SourceT has acceleration dimensions in the incompressible U equation.
// Therefore the integral is thrust/rho = 0.5*Ct*A*U_ref^2, without density.
// Using MODEL area and GLOBAL selected MODEL volume also gives the required
// acceleration scaling a_model = a_physical/scale when U is unchanged.
inline double thrustPerDensity
(
    const double ct, const double area, const double referenceVelocity
)
{
    return 0.5 * ct * area * referenceVelocity * referenceVelocity;
}

inline double axialAcceleration
(
    const double ct, const double area, const double referenceVelocity,
    const double globalSelectedVolume
)
{
    return thrustPerDensity(ct, area, referenceVelocity) / globalSelectedVolume;
}
}

#endif
