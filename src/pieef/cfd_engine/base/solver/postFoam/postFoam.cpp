/*---------------------------------------------------------------------------*\
  Read-only velocity/pressure sampling at vertical heights above the actual
  bot patch. OpenCFD OpenFOAM v2212. No assumptions about cell ordering,
  extrusion layers or a first-layer-derived ground elevation are required.
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "meshSearch.H"
#include "interpolationCellPoint.H"
#include "cJSON.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>


cJSON* required(cJSON* parent, const char* key)
{
    cJSON* item = parent ? cJSON_GetObjectItem(parent, key) : nullptr;
    if (!item)
    {
        std::cerr << "[FATAL] Missing sampling input: " << key << '\n';
        std::exit(EXIT_FAILURE);
    }
    return item;
}


double positiveNumber(cJSON* item, const char* name)
{
    if (!cJSON_IsNumber(item) || !std::isfinite(item->valuedouble) || item->valuedouble <= 0)
    {
        std::cerr << "[FATAL] " << name << " must be a finite positive number\n";
        std::exit(EXIT_FAILURE);
    }
    return item->valuedouble;
}


std::string heightName(double height)
{
    std::ostringstream name;
    name << std::setprecision(15) << height;
    return name.str();
}


void writeSample(std::ostream& output, const point& target, const vector& velocity, scalar pressure)
{
    output << target.x() << '\t' << target.y() << '\t' << target.z() << '\t'
        << velocity.x() << '\t' << velocity.y() << '\t' << velocity.z() << '\t'
        << pressure << '\n';
}


int main(int argc, char* argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    if (Pstream::parRun())
    {
        std::cerr << "[FATAL] postFoam samples the reconstructed serial mesh\n";
        return EXIT_FAILURE;
    }
    volVectorField U
    (
        IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );
    volScalarField p
    (
        IOobject("p", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );

    const label groundPatchId = mesh.boundaryMesh().findPatchID("bot");
    if (groundPatchId < 0 || mesh.boundaryMesh()[groundPatchId].empty())
    {
        std::cerr << "[FATAL] A non-empty bot patch is required for vertical AGL sampling\n";
        return EXIT_FAILURE;
    }
    const polyPatch& groundPatch = mesh.boundaryMesh()[groundPatchId];
    const vectorField groundCentres(groundPatch.faceCentres());
    const labelUList& groundCells = groundPatch.faceCells();

    std::ifstream input("./Input/input.json");
    if (!input)
    {
        std::cerr << "[FATAL] Cannot open ./Input/input.json\n";
        return EXIT_FAILURE;
    }
    const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    cJSON* root = cJSON_Parse(json.c_str());
    if (!root)
    {
        std::cerr << "[FATAL] Invalid sampling input JSON\n";
        return EXIT_FAILURE;
    }
    const double scale = positiveNumber(required(required(root, "mesh"), "scale"), "mesh.scale");
    cJSON* post = required(root, "post");
    cJSON* countItem = required(post, "num_udh");
    const double count = positiveNumber(countItem, "post.num_udh");
    if (count > 1000 || count != std::floor(count))
    {
        std::cerr << "[FATAL] post.num_udh must be an integer in 1..1000\n";
        cJSON_Delete(root);
        return EXIT_FAILURE;
    }
    const int nHeights = static_cast<int>(count);
    cJSON* heightInput = required(post, "udh");
    const bool explicitHeights = cJSON_IsArray(heightInput);
    if (explicitHeights && cJSON_GetArraySize(heightInput) != nHeights)
    {
        std::cerr << "[FATAL] post.num_udh does not match post.udh array length\n";
        cJSON_Delete(root);
        return EXIT_FAILURE;
    }
    std::vector<double> heights(nHeights);
    for (int i = 0; i < nHeights; ++i)
    {
        heights[i] = explicitHeights
            ? positiveNumber(cJSON_GetArrayItem(heightInput, i), "post.udh[]")
            : (i + 1) * positiveNumber(heightInput, "post.udh");
        if (i && heights[i] <= heights[i - 1])
        {
            std::cerr << "[FATAL] Sampling heights must be strictly increasing\n";
            cJSON_Delete(root);
            return EXIT_FAILURE;
        }
    }
    cJSON_Delete(root);

    // findCell returns -1 outside the actual mesh; a nearest cell must never
    // substitute for a missing point. Both fields use the exact same point.
    meshSearch search(mesh, polyMesh::CELL_TETS);
    interpolationCellPoint<vector> interpolateU(U);
    interpolationCellPoint<scalar> interpolateP(p);
    const scalar missing = std::numeric_limits<scalar>::quiet_NaN();
    const vector missingVelocity(missing, missing, missing);
    std::ofstream quality("./agl-sampling-quality.json");
    if (!quality)
    {
        std::cerr << "[FATAL] Cannot write AGL sampling quality metadata\n";
        return EXIT_FAILURE;
    }
    quality << std::setprecision(15)
        << "{\"schema_version\":1,\"method\":\"interpolationCellPoint_at_vertical_AGL\","
        << "\"height_definition\":\"vertical_distance_above_bot_face_centre\","
        << "\"ground_patch\":\"bot\",\"mesh_scale\":" << scale << ",\"layers\":[";

    for (int heighti = 0; heighti < nHeights; ++heighti)
    {
        const double height = heights[heighti];
        std::ofstream output("./Output/plt/" + heightName(height));
        if (!output)
        {
            std::cerr << "[FATAL] Cannot open AGL sample output for " << height << " m\n";
            return EXIT_FAILURE;
        }
        output << std::setprecision(15);
        label valid = 0, outside = 0, nonFinite = 0;
        forAll(groundCentres, facei)
        {
            point target(groundCentres[facei]);
            target.z() += height * scale;
            const label celli = search.findCell(target, groundCells[facei], true);
            if (celli < 0)
            {
                ++outside;
                writeSample(output, target, missingVelocity, missing);
                continue;
            }
            const vector velocity = interpolateU.interpolate(target, celli);
            const scalar pressure = interpolateP.interpolate(target, celli);
            if (!std::isfinite(velocity.x()) || !std::isfinite(velocity.y())
                || !std::isfinite(velocity.z()) || !std::isfinite(pressure))
            {
                ++nonFinite;
                writeSample(output, target, missingVelocity, missing);
                continue;
            }
            ++valid;
            writeSample(output, target, velocity, pressure);
        }
        output.close();
        if (!output)
        {
            std::cerr << "[FATAL] Failed to write complete AGL samples\n";
            return EXIT_FAILURE;
        }
        if (heighti) quality << ',';
        quality << "{\"height_m\":" << height << ",\"requested_point_count\":" << groundCentres.size()
            << ",\"valid_point_count\":" << valid << ",\"outside_domain_count\":" << outside
            << ",\"non_finite_count\":" << nonFinite << '}';
        Info << "AGL sampling " << height << " m: valid=" << valid << '/' << groundCentres.size()
             << ", outside_domain=" << outside << ", non_finite=" << nonFinite << nl;
    }
    quality << "]}\n";
    quality.close();
    if (!quality)
    {
        std::cerr << "[FATAL] Failed to write AGL sampling quality metadata\n";
        return EXIT_FAILURE;
    }
    Info << "Read-only vertical AGL sampling complete.\nEnd\n" << endl;
    return 0;
}
