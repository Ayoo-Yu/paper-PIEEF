#ifndef WIND_TERRAIN_GEOMETRY_H
#define WIND_TERRAIN_GEOMETRY_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace windTerrain
{
static const char anchorVersion[] = "vertical-bottom-triangle-v1";
typedef std::array<double, 3> Point;
typedef std::array<Point, 3> Triangle;

struct TriangleHit
{
    bool found;
    double z;
    std::array<double, 3> barycentric;
};

// Intersect a vertical line through the exact query XY with a triangle.
// This is a surface intersection, not nearest-centre elevation interpolation.
// Reversed winding and shared edges/vertices are valid. A vertical/degenerate
// XY projection cannot define a single terrain height and produces no hit.
inline TriangleHit verticalHit(const Triangle& p, double x, double y)
{
    TriangleHit hit = {false, 0.0, {{0.0, 0.0, 0.0}}};
    const double ux = p[1][0] - p[0][0], uy = p[1][1] - p[0][1];
    const double vx = p[2][0] - p[0][0], vy = p[2][1] - p[0][1];
    const double qx = x - p[0][0], qy = y - p[0][1];
    const double determinant = ux * vy - uy * vx;
    const double extent2 = std::max(ux*ux + uy*uy, vx*vx + vy*vy);
    if (!std::isfinite(determinant) || !std::isfinite(qx) || !std::isfinite(qy)
        || std::abs(determinant) <= 64 * std::numeric_limits<double>::epsilon() * extent2)
    {
        return hit;
    }
    const double b = (qx * vy - qy * vx) / determinant;
    const double c = (ux * qy - uy * qx) / determinant;
    const double a = 1.0 - b - c;
    const double tolerance = 1.0e-10;
    if (a < -tolerance || b < -tolerance || c < -tolerance
        || a > 1 + tolerance || b > 1 + tolerance || c > 1 + tolerance)
    {
        return hit;
    }
    // Clamp only round-off at a shared edge/vertex, never extrapolate a missing
    // surface. The returned XY remains the caller's exact turbine location.
    hit.barycentric = {{std::max(0.0, a), std::max(0.0, b), std::max(0.0, c)}};
    const double sum = hit.barycentric[0] + hit.barycentric[1] + hit.barycentric[2];
    for (int i = 0; i < 3; ++i)
    {
        hit.barycentric[i] /= sum;
        hit.z += hit.barycentric[i] * p[i][2];
    }
    hit.found = std::isfinite(hit.z);
    return hit;
}
}

// The geometry kernel above can also be compiled without OpenFOAM. The native
// solver and native MPI regression both use this exact surface/MPI adapter.
#ifdef WIND_TERRAIN_OPENFOAM
#include "fvMesh.H"
#include "PstreamReduceOps.H"

namespace windTerrain
{
struct BottomTriangle
{
    Triangle points;
    Foam::label patchFace;
    Foam::label triangleIndex;
    double xmin, xmax, ymin, ymax;
};

class BottomSurface
{
    std::vector<BottomTriangle> triangles_;

public:
    BottomSurface(const Foam::fvMesh& mesh, Foam::label patchID)
    {
        const Foam::polyPatch& patch = mesh.boundaryMesh()[patchID];
        const Foam::pointField& points = mesh.points();
        forAll(patch, faceI)
        {
            const Foam::face& polygon = mesh.faces()[patch.start() + faceI];
            Foam::faceList triangulated(polygon.nTriangles());
            Foam::label nextTriangle = 0;
            // v2212 splits polygon faces using their actual existing vertices;
            // this also handles concave polygons without a face-centre fan.
            polygon.triangles(points, nextTriangle, triangulated);
            for (Foam::label triI = 0; triI < nextTriangle; ++triI)
            {
                BottomTriangle triangle;
                triangle.patchFace = faceI;
                triangle.triangleIndex = triI;
                for (int pI = 0; pI < 3; ++pI)
                {
                    const Foam::point& p = points[triangulated[triI][pI]];
                    triangle.points[pI] = {{p.x(), p.y(), p.z()}};
                }
                triangle.xmin = triangle.xmax = triangle.points[0][0];
                triangle.ymin = triangle.ymax = triangle.points[0][1];
                for (int pI = 1; pI < 3; ++pI)
                {
                    triangle.xmin = std::min(triangle.xmin, triangle.points[pI][0]);
                    triangle.xmax = std::max(triangle.xmax, triangle.points[pI][0]);
                    triangle.ymin = std::min(triangle.ymin, triangle.points[pI][1]);
                    triangle.ymax = std::max(triangle.ymax, triangle.points[pI][1]);
                }
                triangles_.push_back(triangle);
            }
        }
    }

    const std::vector<BottomTriangle>& triangles() const { return triangles_; }
};

struct TerrainAnchor
{
    bool found;
    bool ambiguous;
    double z;
    double heightSpread;
    Foam::label owner;
    Foam::label patchFace;
    Foam::label triangleIndex;
    Foam::label globalHitCount;
};

// Collective on all ranks. At a shared edge/point several valid triangles may
// intersect the vertical line. Reduce height once, then choose the lowest rank,
// lowest local face and lowest triangle index among equal-height hits. Never
// sum elevations. A missing or vertically ambiguous terrain surface is explicit.
inline TerrainAnchor findAnchor(const BottomSurface& surface, double x, double y)
{
    struct Candidate { double z; Foam::label face, triangle; };
    std::vector<Candidate> candidates;
    double localMax = -Foam::GREAT, localMin = Foam::GREAT;
    for (const BottomTriangle& triangle : surface.triangles())
    {
        const double boxTolerance = 1.0e-10 * std::max(
            triangle.xmax - triangle.xmin, triangle.ymax - triangle.ymin);
        if (x < triangle.xmin - boxTolerance || x > triangle.xmax + boxTolerance
            || y < triangle.ymin - boxTolerance || y > triangle.ymax + boxTolerance)
        {
            continue;
        }
        const TriangleHit hit = verticalHit(triangle.points, x, y);
        if (!hit.found) continue;
        candidates.push_back({hit.z, triangle.patchFace, triangle.triangleIndex});
        localMax = std::max(localMax, hit.z);
        localMin = std::min(localMin, hit.z);
    }
    TerrainAnchor result = {false, false, 0.0, 0.0, -1, -1, -1, 0};
    result.globalHitCount = Foam::returnReduce(
        Foam::label(candidates.size()), Foam::sumOp<Foam::label>());
    const double globalMax = Foam::returnReduce(localMax, Foam::maxOp<double>());
    const double globalMin = Foam::returnReduce(localMin, Foam::minOp<double>());
    if (!result.globalHitCount) return result;
    const double heightTolerance = 1.0e-10 * std::max(1.0, std::abs(globalMax));
    result.z = globalMax;
    result.heightSpread = globalMax - globalMin;
    result.ambiguous = result.heightSpread > heightTolerance;
    Foam::label localFace = std::numeric_limits<Foam::label>::max();
    Foam::label localTriangle = std::numeric_limits<Foam::label>::max();
    for (const Candidate& candidate : candidates)
    {
        if (std::abs(candidate.z - globalMax) <= heightTolerance
            && (candidate.face < localFace
                || (candidate.face == localFace && candidate.triangle < localTriangle)))
        {
            localFace = candidate.face;
            localTriangle = candidate.triangle;
        }
    }
    Foam::label owner = localFace != std::numeric_limits<Foam::label>::max()
        ? Foam::Pstream::myProcNo() : Foam::Pstream::nProcs();
    result.owner = Foam::returnReduce(owner, Foam::minOp<Foam::label>());
    if (Foam::Pstream::myProcNo() != result.owner)
    {
        localFace = localTriangle = std::numeric_limits<Foam::label>::max();
    }
    result.patchFace = Foam::returnReduce(localFace, Foam::minOp<Foam::label>());
    result.triangleIndex = Foam::returnReduce(localTriangle, Foam::minOp<Foam::label>());
    result.found = result.owner < Foam::Pstream::nProcs();
    return result;
}
}
#endif
#endif
