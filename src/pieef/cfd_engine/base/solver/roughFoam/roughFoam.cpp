/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    simpleFoam  // with vegetation-roughness body-force support

Group
    grpIncompressibleSolvers

Description
    Steady-state solver for incompressible, turbulent flows, extended with
    vegetation roughness body force constructed from a custom ASCII file “Input/rou”.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

#include "cJSON.h"
#include "actuatorMath.H"
#define WIND_TERRAIN_OPENFOAM
#include "terrainGeometry.H"

// Debug level: 0 summary, 1 more logs, 2 also write large debug files
static int dbgRoughLevel = 0;

// --------------------- Global variables --------------------- //
// some variables about adm
FILE *file0 = nullptr, *file1 = nullptr;
static long int i, j, k;
static int numTurbine, numUPCt[10] = {0}, botID = 6, numType;
static int onNode[200] = {0}, numU[200] = {0}, numDisk[200] = {0};
static double XY[200][5] = {0}, UPCt[10][200][3] = {0}, realXYZ[200][3] = {0}, realHigh[200] = {0};
static double sumU[200] = {0}, windU[200] = {0};
static double sampleVolume[200] = {0}, diskVolume[200] = {0};
static double Power[200] = {0}, Ct[200] = {0}, fn[200] = {0};
static double angle, vInlet, scale, dx[200] = {0}, botHigh;
static const double pi = 3.141592654, p_air = 1.225;
static std::vector<Foam::label> sampleCells[200], diskCells[200];

// some variables about roughness
std::ifstream in;
static double lat, lon, utmX = 0.0, utmY = 0.0, cee = 100.0;
static double lt, h, h1, ceng, q1, r1, r2, r12, Cd, lad_max, vege_times;

// --------------------- Utilities --------------------- //
static inline int utmZoneFromLon(double lonDeg)
{
    return int(std::floor(lonDeg/6.0) + 31.0);
}

static inline double getJsonDouble(cJSON* obj, const char* key)
{
    cJSON* it = cJSON_GetObjectItem(obj, key);
    if (!it || !cJSON_IsNumber(it))
    {
        FatalErrorInFunction << "Missing or non-numeric JSON key: " << key << exit(FatalError);
    }
    return it->valuedouble;
}

static FILE* fopenOrDie(const char* path, const char* mode)
{
    FILE* f = std::fopen(path, mode);
    if (!f)
    {
        FatalErrorInFunction << "Cannot open file: " << path << exit(FatalError);
    }
    return f;
}

static void read_Turbines()
{
    file0 = fopenOrDie("Input/Turbines.txt", "r");

    // Count lines
    numTurbine = 0;
    char line[1024];
    while (std::fgets(line, sizeof(line), file0)) { numTurbine++; }
    std::fclose(file0);

    if (numTurbine <= 0)
    {
        numTurbine = 0;
        Info<< "No turbines configured (Input/Turbines.txt is empty). Running terrain/roughness-only simulation." << nl;
        return;
    }
    if (numTurbine > 200)
    {
        FatalErrorInFunction << "Too many turbines (" << numTurbine << "), max=200." << exit(FatalError);
    }

    file0 = fopenOrDie("Input/Turbines.txt", "r");
    for (i = 0; i < numTurbine; i++)
    {
        if (std::fscanf(file0, "%lf%lf%lf%lf%lf", &XY[i][0], &XY[i][1], &XY[i][2], &XY[i][3], &XY[i][4]) != 5)
        {
            std::fclose(file0);
            FatalErrorInFunction << "Bad line format in Input/Turbines.txt at row " << (i+1) << exit(FatalError);
        }
        for (j = 0; j < 5; ++j)
        {
            if (!std::isfinite(XY[i][j]))
            {
                FatalErrorInFunction << "Non-finite turbine input at row " << (i+1) << exit(FatalError);
            }
        }
        if (XY[i][2] <= 0 || XY[i][3] <= 0 || XY[i][4] < 1
            || XY[i][4] > 10 || XY[i][4] != std::floor(XY[i][4]))
        {
            FatalErrorInFunction << "Invalid hub height, rotor diameter or curve type at row " << (i+1) << exit(FatalError);
        }
    }
    std::fclose(file0);
}

static void read_InputJSON()
{
    // numType from turbines (types are in XY[][4]) - optional for pure flow runs
    if (numTurbine > 0)
    {
        numType = int(XY[0][4]);
        for (i = 0; i < numTurbine; i++) if (XY[i][4] > numType) numType = int(XY[i][4]);
        if (numType <= 0 || numType > 10)
        {
            FatalErrorInFunction << "Invalid numType=" << numType << " (expected 1..10)." << exit(FatalError);
        }
    }
    else
    {
        numType = 0;
    }

    file0 = fopenOrDie("Input/input.json", "r");
    std::fseek(file0, 0, SEEK_END);
    long file_size = std::ftell(file0);
    std::fseek(file0, 0, SEEK_SET);

    if (file_size <= 0)
    {
        std::fclose(file0);
        FatalErrorInFunction << "Input/input.json is empty." << exit(FatalError);
    }

    char *json_str = (char*)std::malloc(size_t(file_size) + 1);
    if (!json_str)
    {
        std::fclose(file0);
        FatalErrorInFunction << "malloc failed for JSON buffer." << exit(FatalError);
    }

    size_t nread = std::fread(json_str, 1, size_t(file_size), file0);
    std::fclose(file0);
    if (nread != size_t(file_size))
    {
        std::free(json_str);
        FatalErrorInFunction << "Failed to read Input/input.json fully." << exit(FatalError);
    }
    json_str[file_size] = '\0'; // critical: null-terminate for cJSON

    cJSON *cjson = cJSON_Parse(json_str);
    if (!cjson)
    {
        std::free(json_str);
        FatalErrorInFunction << "cJSON_Parse failed. Check Input/input.json syntax." << exit(FatalError);
    }

    // domain
    cJSON *cjson_domain = cJSON_GetObjectItem(cjson, "domain");
    if (!cjson_domain) { std::free(json_str); cJSON_Delete(cjson);
        FatalErrorInFunction << "Missing 'domain' object in input.json." << exit(FatalError); }

    lon = getJsonDouble(cjson_domain, "centerLon");
    lat = getJsonDouble(cjson_domain, "centerLat");
    lt  = getJsonDouble(cjson_domain, "lt");
    h   = getJsonDouble(cjson_domain, "h");

    // wind
    cJSON *cjson_wind = cJSON_GetObjectItem(cjson, "wind");
    if (!cjson_wind) { std::free(json_str); cJSON_Delete(cjson);
        FatalErrorInFunction << "Missing 'wind' object in input.json." << exit(FatalError); }

    angle  = getJsonDouble(cjson_wind, "angle");
    angle  = -(90.0 + angle) * pi / 180.0;
    vInlet = getJsonDouble(cjson_wind, "speed");

    // mesh
    cJSON *cjson_mesh = cJSON_GetObjectItem(cjson, "mesh");
    if (!cjson_mesh) { std::free(json_str); cJSON_Delete(cjson);
        FatalErrorInFunction << "Missing 'mesh' object in input.json." << exit(FatalError); }

    h1    = getJsonDouble(cjson_mesh, "h1");
    ceng  = getJsonDouble(cjson_mesh, "ceng");
    q1    = getJsonDouble(cjson_mesh, "q1");
    scale = getJsonDouble(cjson_mesh, "scale");
    botHigh = scale * h1 / ceng;

    cJSON *cjson_mesh_lc2 = cJSON_GetObjectItem(cjson_mesh, "lc2");
    if (numTurbine > 0)
    {
        if (!cjson_mesh_lc2 || !cJSON_IsArray(cjson_mesh_lc2))
        {
            std::free(json_str); cJSON_Delete(cjson);
            FatalErrorInFunction << "mesh.lc2 must be an array." << exit(FatalError);
        }

        int lc2Size = cJSON_GetArraySize(cjson_mesh_lc2);
        if (lc2Size < numType)
        {
            std::free(json_str); cJSON_Delete(cjson);
            FatalErrorInFunction << "mesh.lc2 size (" << lc2Size << ") < numType (" << numType << ")." << exit(FatalError);
        }

        for (i = 0; i < numTurbine; i++)
        {
            k = int(XY[i][4]) - 1;
            cJSON *cjson_dx = cJSON_GetArrayItem(cjson_mesh_lc2, k);
            if (!cjson_dx || !cJSON_IsNumber(cjson_dx))
            {
                std::free(json_str); cJSON_Delete(cjson);
                FatalErrorInFunction << "mesh.lc2[" << k << "] is missing or not a number." << exit(FatalError);
            }
            dx[i] = scale * cjson_dx->valuedouble;
        }
    }

    // terrain radii
    cJSON *cjson_terrain = cJSON_GetObjectItem(cjson, "terrain");
    if (!cjson_terrain) { std::free(json_str); cJSON_Delete(cjson);
        FatalErrorInFunction << "Missing 'terrain' object in input.json." << exit(FatalError); }

    r1 = getJsonDouble(cjson_terrain, "r1");
    r2 = getJsonDouble(cjson_terrain, "r2");
    r12 = 0.5*(r1 + r2);

    // roughness params
    cJSON *cjson_roughness = cJSON_GetObjectItem(cjson, "roughness");
    if (!cjson_roughness) { std::free(json_str); cJSON_Delete(cjson);
        FatalErrorInFunction << "Missing 'roughness' object in input.json." << exit(FatalError); }

    Cd         = getJsonDouble(cjson_roughness, "Cd");
    lad_max    = getJsonDouble(cjson_roughness, "lad_max");
    vege_times = getJsonDouble(cjson_roughness, "vege_times");

    std::free(json_str);
    cJSON_Delete(cjson);

    // scale XY and rotate
    for (i = 0; i < numTurbine; i++)
    {
        for (j = 0; j < 4; j++) XY[i][j] *= scale;
    }
    if (Pstream::master()) file1 = fopenOrDie("Output/Output01-XY", "w");
    for (i = 0; i < numTurbine; i++)
    {
        double tempX = XY[i][0], tempY = XY[i][1];
        XY[i][0] = tempX * Foam::cos(angle) + tempY * Foam::sin(angle);
        XY[i][1] = -tempX * Foam::sin(angle) + tempY * Foam::cos(angle);
        const double halfDomain = lt * scale * 0.5;
        const double radius = XY[i][3] * 0.5;
        const double diskHalfWidth = windPhysics::diskHalfThicknessRatio * XY[i][3];
        const double referenceCentre = XY[i][0] - windPhysics::referenceDistanceRatio * XY[i][3];
        const double referenceHalfWidth = windPhysics::referenceHalfThicknessRatio * XY[i][3];
        const double upstreamMin = referenceCentre - referenceHalfWidth;
        const double upstreamMax = referenceCentre + referenceHalfWidth;
        if (std::fabs(XY[i][0]) + diskHalfWidth >= halfDomain
            || std::fabs(XY[i][1]) + radius >= halfDomain
            || upstreamMin <= -halfDomain || upstreamMax >= halfDomain)
        {
            FatalErrorInFunction << "Turbine " << (i + 1)
                << " or its upstream sampling window lies outside the rotated domain; enlarge domain.lt."
                << exit(FatalError);
        }
        if (Pstream::master()) std::fprintf(file1, "%f\t%f\t%f\t%f\t%f\n", XY[i][0], XY[i][1], XY[i][2], XY[i][3], XY[i][4]);
    }
    if (Pstream::master()) std::fclose(file1);

    if (numTurbine > 0)
    {
        Info<< "angle(rad)=" << angle
            << " vInlet=" << vInlet
            << " scale=" << scale
            << " dx[0]=" << dx[0] << nl;
    }
    else
    {
        Info<< "angle(rad)=" << angle
            << " vInlet=" << vInlet
            << " scale=" << scale << nl;
    }
}

static char *numToStr(int num, char *str, int radix)
{
    char index[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    unsigned unum;
    int i = 0, j, k;
    if (radix == 10 && num < 0) { unum = (unsigned)-num; str[i++] = '-'; }
    else unum = (unsigned)num;
    do { str[i++] = index[unum % (unsigned)radix]; unum /= radix; } while (unum);
    str[i] = '\0';
    k = (str[0] == '-') ? 1 : 0;
    char temp;
    for (j = k; j <= (i - 1) / 2; j++) { temp = str[j]; str[j] = str[i - 1 + k - j]; str[i - 1 + k - j] = temp; }
    return str;
}

static void read_U_P_Ct()
{
    char strPath[64], strNum[32], line[1024];
    for (i = 0; i < numType; i++)
    {
        std::strcpy(strPath, "Input/");
        numToStr((int)i + 1, strNum, 10);
        std::strcat(strPath, std::strcat(strNum, "-U-P-Ct.txt"));

        file0 = fopenOrDie(strPath, "r");
        numUPCt[i] = 0;
        while (std::fgets(line, sizeof(line), file0)) { numUPCt[i]++; }
        std::fclose(file0);

        if (numUPCt[i] <= 1)
        {
            FatalErrorInFunction << strPath << " has too few rows." << exit(FatalError);
        }
        if (numUPCt[i] > 200)
        {
            FatalErrorInFunction << strPath << " has too many rows (>200)." << exit(FatalError);
        }

        file0 = fopenOrDie(strPath, "r");
        for (j = 0; j < numUPCt[i]; j++)
        {
            if (std::fscanf(file0, "%lf%lf%lf\n", &UPCt[i][j][0], &UPCt[i][j][1], &UPCt[i][j][2]) != 3)
            {
                std::fclose(file0);
                FatalErrorInFunction << "Bad line format in " << strPath << " at row " << (j+1) << exit(FatalError);
            }
            if (!std::isfinite(UPCt[i][j][0]) || !std::isfinite(UPCt[i][j][1])
                || !std::isfinite(UPCt[i][j][2]) || UPCt[i][j][0] < 0
                || UPCt[i][j][1] < 0 || UPCt[i][j][2] < 0
                || (j > 0 && UPCt[i][j][0] <= UPCt[i][j-1][0]))
            {
                FatalErrorInFunction << "Curve values must be finite/non-negative with strictly increasing wind speeds: "
                    << strPath << " row " << (j+1) << exit(FatalError);
            }
        }
        std::fclose(file0);
    }
}

static void update_vInlet()
{
    // Keep the inlet velocity configured in input.json.
    // Overwriting it with the first turbine's sampled local wind speed makes
    // the solution order-dependent and rounds away fractional inlet speeds.
    Info << "angle=" << angle << " keeping configured vInlet=" << vInlet
         << " scale=" << scale;
    if (numTurbine > 0)
    {
        Info << " firstWindU=" << windU[0] << " dx[0]=" << dx[0];
    }
    Info << endl;
}

// The reference cylinder is centred 1D upstream, radius D/4, half-length
// D/4. It supplies the same free-stream-style Ct lookup as the legacy model;
// it is NOT the disk velocity used by a Ct-prime induction model.
static void buildActuatorCellSets(const fvMesh& mesh)
{
    for (int turbine = 0; turbine < numTurbine; ++turbine)
    {
        const double diameter = XY[turbine][3];
        double localSampleVolume = 0.0, localDiskVolume = 0.0;
        sampleCells[turbine].clear();
        diskCells[turbine].clear();
        forAll(mesh.cells(), cellI)
        {
            const vector& centre = mesh.C()[cellI];
            const double volume = mesh.V()[cellI];
            if (!std::isfinite(volume) || volume <= 0.0)
            {
                FatalErrorInFunction << "Invalid cell volume at cell " << cellI << exit(FatalError);
            }
            if (windPhysics::insideAxialCylinder
                (centre.x(), centre.y(), centre.z(),
                 XY[turbine][0] - windPhysics::referenceDistanceRatio * diameter,
                 XY[turbine][1], realHigh[turbine],
                 windPhysics::referenceHalfThicknessRatio * diameter,
                 windPhysics::referenceRadiusRatio * diameter))
            {
                sampleCells[turbine].push_back(cellI);
                localSampleVolume += volume;
            }
            if (windPhysics::insideAxialCylinder
                (centre.x(), centre.y(), centre.z(),
                 XY[turbine][0], XY[turbine][1], realHigh[turbine],
                 windPhysics::diskHalfThicknessRatio * diameter, diameter * 0.5))
            {
                diskCells[turbine].push_back(cellI);
                localDiskVolume += volume;
            }
        }
        numU[turbine] = returnReduce(int(sampleCells[turbine].size()), sumOp<int>());
        numDisk[turbine] = returnReduce(int(diskCells[turbine].size()), sumOp<int>());
        sampleVolume[turbine] = returnReduce(localSampleVolume, sumOp<double>());
        diskVolume[turbine] = returnReduce(localDiskVolume, sumOp<double>());
        if (numU[turbine] == 0 || !std::isfinite(sampleVolume[turbine]) || sampleVolume[turbine] <= 0.0)
        {
            FatalErrorInFunction << "No valid upstream reference volume for turbine " << (turbine+1)
                << "; refine the mesh or move the reference window inside the domain."
                << exit(FatalError);
        }
        if (numDisk[turbine] == 0 || !std::isfinite(diskVolume[turbine]) || diskVolume[turbine] <= 0.0)
        {
            FatalErrorInFunction << "No valid actuator volume for turbine " << (turbine+1)
                << "; refine the mesh instead of silently omitting its thrust." << exit(FatalError);
        }
    }
    Info << "Actuator reference: volume-weighted signed Ux, cylinder at -1D, radius D/4, half-length D/4."
         << nl << "Actuator support: radius D/2, half-length D/4, globally volume-normalized acceleration (no density)."
         << nl;
}

static void sampleReferenceVelocity(const fvMesh& mesh, const volVectorField& U)
{
    for (int turbine = 0; turbine < numTurbine; ++turbine)
    {
        double localSum = 0.0, localVelocityVolume = 0.0;
        for (const label cellI : sampleCells[turbine])
        {
            const double velocity = U[cellI].x();
            if (!std::isfinite(velocity))
            {
                FatalErrorInFunction << "Non-finite reference velocity for turbine " << (turbine+1)
                    << " at cell " << cellI << exit(FatalError);
            }
            localSum += velocity;
            localVelocityVolume += velocity * mesh.V()[cellI];
        }
        // Preserve Output05's legacy sum(Ux) column; the third column is now
        // the volume-weighted reference velocity, not sum(Ux)/cell-count.
        sumU[turbine] = returnReduce(localSum, sumOp<double>());
        const double velocityVolume = returnReduce(localVelocityVolume, sumOp<double>());
        windU[turbine] = windPhysics::referenceMean(velocityVolume, sampleVolume[turbine]);
        if (!std::isfinite(windU[turbine]))
        {
            FatalErrorInFunction << "Invalid reference mean for turbine " << (turbine+1) << exit(FatalError);
        }
    }
}

static void updateTurbineOperatingPoints()
{
    for (int turbine = 0; turbine < numTurbine; ++turbine)
    {
        const int type = int(XY[turbine][4]) - 1;
        Power[turbine] = 0.0;
        Ct[turbine] = 0.0;
        if (windU[turbine] == UPCt[type][numUPCt[type]-1][0])
        {
            Power[turbine] = UPCt[type][numUPCt[type]-1][1];
            Ct[turbine] = UPCt[type][numUPCt[type]-1][2];
        }
        else
        {
            for (int row = 0; row+1 < numUPCt[type]; ++row)
            {
                if (windU[turbine] >= UPCt[type][row][0] && windU[turbine] < UPCt[type][row+1][0])
                {
                    const double fraction = (windU[turbine] - UPCt[type][row][0])
                        / (UPCt[type][row+1][0] - UPCt[type][row][0]);
                    Power[turbine] = UPCt[type][row][1] + fraction * (UPCt[type][row+1][1] - UPCt[type][row][1]);
                    Ct[turbine] = UPCt[type][row][2] + fraction * (UPCt[type][row+1][2] - UPCt[type][row][2]);
                    break;
                }
            }
        }
        const double area = pi * XY[turbine][3] * XY[turbine][3] * 0.25;
        fn[turbine] = windPhysics::axialAcceleration(Ct[turbine], area, windU[turbine], diskVolume[turbine]);
        if (!std::isfinite(fn[turbine]) || fn[turbine] < 0.0)
        {
            FatalErrorInFunction << "Invalid actuator acceleration for turbine " << (turbine+1) << exit(FatalError);
        }
    }
}

static void applyActuatorSources(volVectorField& SourceT)
{
    SourceT *= 0.0;
    for (int turbine = 0; turbine < numTurbine; ++turbine)
    {
        for (const label cellI : diskCells[turbine])
        {
            // Accumulate if supports overlap; assignment would lose thrust.
            SourceT[cellI].x() -= fn[turbine];
        }
    }
}

static void writeReferenceSamples(const char* path)
{
    if (!Pstream::master()) return;
    FILE* out = fopenOrDie(path, "w");
    for (int turbine = 0; turbine < numTurbine; ++turbine)
    {
        std::fprintf(out, "%.12g\t%d\t%.12g\n", sumU[turbine], numU[turbine], windU[turbine]);
    }
    std::fclose(out);
}

static void writeOperatingPoints(const char* path)
{
    if (!Pstream::master()) return;
    FILE* out = fopenOrDie(path, "w");
    for (int turbine = 0; turbine < numTurbine; ++turbine)
    {
        std::fprintf(out, "%.12g\t%.12g\t%.12g\t%.12g\n", windU[turbine], Power[turbine], Ct[turbine], fn[turbine]);
    }
    std::fclose(out);
}

static void writeActuatorDiagnostics(const fvMesh& mesh, const char* stage, const double iteration, const bool initialise = false)
{
    // Integral of EACH turbine's discrete contribution, before superposition.
    // Final-resampled entries describe the final operating point; that source
    // has not necessarily been used by a subsequent momentum solve.
    double discreteIntegral[200] = {0};
    for (int turbine = 0; turbine < numTurbine; ++turbine)
    {
        double localIntegral = 0.0;
        for (const label cellI : diskCells[turbine]) localIntegral += fn[turbine] * mesh.V()[cellI];
        discreteIntegral[turbine] = returnReduce(localIntegral, sumOp<double>());
    }
    if (!Pstream::master()) return;
    FILE* out = fopenOrDie("Output/actuator-diagnostics.tsv", initialise ? "w" : "a");
    if (initialise)
    {
        std::fprintf(out, "stage\titeration\tturbine\tupstream_cells\tupstream_volume_model\tdisk_cells\tdisk_volume_model\twind_speed_m_s\tct\taccel_integral_target_model\taccel_integral_discretized_model\n");
    }
    for (int turbine = 0; turbine < numTurbine; ++turbine)
    {
        const double area = pi * XY[turbine][3] * XY[turbine][3] * 0.25;
        const double target = windPhysics::thrustPerDensity(Ct[turbine], area, windU[turbine]);
        std::fprintf(out, "%s\t%.12g\t%d\t%d\t%.12g\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n",
            stage, iteration, turbine+1, numU[turbine], sampleVolume[turbine], numDisk[turbine], diskVolume[turbine],
            windU[turbine], Ct[turbine], target, discreteIntegral[turbine]);
    }
    std::fclose(out);
}

static void LonLat2UTM(double longitude, double latitude, double &UTME, double &UTMN)
{
    double DEG_TO_RAD_LOCAL = 3.14159265358979323846 / 180.0;
    double lon = longitude, lat = latitude;

    // WGS84; 'a' in km, convert to m at end
    double a = 6378.137;
    double e = 0.0818192;
    double k0 = 0.9996;
    double E0 = 500;
    double N0 = 0;

    double zoneNumber = std::floor(lon / 6.0) + 31.0;
    double lambda0 = ((zoneNumber - 1.0) * 6.0 - 180.0 + 3.0) * DEG_TO_RAD_LOCAL;
    double phi = lat * DEG_TO_RAD_LOCAL;
    double lambda = lon * DEG_TO_RAD_LOCAL;

    double v = 1.0 / Foam::sqrt(1.0 - std::pow(e * Foam::sin(phi), 2));
    double A = (lambda - lambda0) * Foam::cos(phi);
    double T = std::pow(Foam::tan(phi), 2);
    double C = std::pow(e, 2) / (1.0 - std::pow(e, 2)) * std::pow(Foam::cos(phi), 2);
    double s = (1 - std::pow(e, 2) / 4 - 3 * std::pow(e, 4) / 64 - 5 * std::pow(e, 6) / 256) * phi
             - (3 * std::pow(e, 2) / 8 + 3 * std::pow(e, 4) / 32 + 45 * std::pow(e, 6) / 1024) * Foam::sin(2 * phi)
             + (15 * std::pow(e, 4) / 256 + 45 * std::pow(e, 6) / 1024) * Foam::sin(4 * phi)
             - 35 * std::pow(e, 6) / 3072 * Foam::sin(6 * phi);

    UTME = E0 + k0 * a * v * (A + (1 - T + C) * std::pow(A, 3) / 6 + (5 - 18 * T + T * T) * std::pow(A, 5) / 120);
    UTMN = N0 + k0 * a * (s + v * Foam::tan(phi) * (std::pow(A, 2) / 2 + (5 - T + 9 * C + 4 * C * C) * std::pow(A, 4) / 24 + (61 - 58 * T + T * T) * std::pow(A, 6) / 720));

    UTME *= 1000.0;
    UTMN *= 1000.0;
}

int main(int argc, char *argv[])
{
    // Optional: --debugRough 0|1|2
    for (int ai=1; ai<argc; ++ai)
    {
        if (std::string(argv[ai]) == "--debugRough" && ai+1 < argc)
            dbgRoughLevel = ::atoi(argv[ai+1]);
    }

#include "setRootCase.H"

    // argList (setRootCase) initialises MPI before any master-only I/O.
    read_Turbines();
    read_InputJSON();
    if (numTurbine > 0)
    {
        read_U_P_Ct();
    }

#include "createTime.H"
#include "createMesh.H"
#include "postProcess.H"
#include "addCheckCaseOptions.H"
#include "createControl.H"
    // Check bottom patch exists early
    botID = mesh.boundaryMesh().findPatchID("bot");
    if (botID < 0)
    {
        if (Pstream::master())
        {
            Info << "Patch 'bot' not found. Available patches:" << nl;
            forAll(mesh.boundaryMesh(), patchI)
            {
                Info << "  - " << mesh.boundaryMesh()[patchI].name() << nl;
            }
        }
        FatalErrorInFunction << "Required patch 'bot' is missing." << exit(FatalError);
    }

#include "createFields.H"
#include "initContinuityErrs.H"

// =========================== calc_RealHigh() ============================= //
#pragma region
    for (i = 0; i < numTurbine; i++)
    {
        onNode[i] = 0;
        realXYZ[i][0] = realXYZ[i][1] = realXYZ[i][2] = 0.0;
        realHigh[i] = 0.0;
    }

    label sumUnits = mesh.boundaryMesh()[botID].size();
    double Dxy[200], tempDxy, botFaceX, botFaceY;
    const vectorField& bottomCentres = mesh.Cf().boundaryField()[botID];
    const windTerrain::BottomSurface bottomSurface(mesh, botID);
    std::vector<windTerrain::TerrainAnchor> terrainAnchors;

    for (i = 0; i < numTurbine; i++)
    {
        Dxy[i] = GREAT;
        forAll(bottomCentres, faceI)
        {
            botFaceX = bottomCentres[faceI].x();
            botFaceY = bottomCentres[faceI].y();
            tempDxy = Foam::sqrt((XY[i][0] - botFaceX)*(XY[i][0] - botFaceX) + (XY[i][1] - botFaceY)*(XY[i][1] - botFaceY));
            if (tempDxy <= Dxy[i]) Dxy[i] = tempDxy;
        }
    }

    Info << "整体网格数量: " << returnReduce(mesh.nCells(), sumOp<label>()) << endl;
    Info << "底层网格数量: " << returnReduce(sumUnits, sumOp<label>()) << endl;
    Info << "Terrain anchor: " << windTerrain::anchorVersion
         << " (actual bottom-triangle vertical intersection at turbine XY)" << endl;

    for (i = 0; i < numTurbine; i++)
    {
        Dxy[i] = returnReduce(Dxy[i], minOp<double>());
        const windTerrain::TerrainAnchor anchor = windTerrain::findAnchor(
            bottomSurface, XY[i][0], XY[i][1]);
        if (!anchor.found || anchor.ambiguous)
        {
            FatalErrorInFunction
                << "No unique vertical bottom-surface intersection at exact XY for turbine "
                << (i + 1) << ": x=" << XY[i][0] << " y=" << XY[i][1]
                << " hits=" << anchor.globalHitCount
                << " heightSpread=" << anchor.heightSpread
                << ". Check terrain coverage and single-valued bottom geometry."
                << exit(FatalError);
        }
        terrainAnchors.push_back(anchor);
        onNode[i] = anchor.owner;
        // findAnchor already returns the same reduced height on every rank.
        // Do not sum these coordinates again: that would multiply by nProcs.
        realXYZ[i][0] = XY[i][0];
        realXYZ[i][1] = XY[i][1];
        realXYZ[i][2] = anchor.z;
        realHigh[i] = XY[i][2] + anchor.z;
    }

    if (Pstream::master()) file1 = fopenOrDie("Output/Output02-realHigh", "w");
    for (i = 0; i < numTurbine; i++)
    {
        // Keep the legacy six-column/model-unit contract. Dxy remains the
        // nearest bottom-face-centre distance; XYZ now names the exact anchor.
        if (Pstream::master()) std::fprintf(file1, "WT-%ld on Node-%d\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n",
            i + 1, onNode[i], Dxy[i], realXYZ[i][0], realXYZ[i][1], realXYZ[i][2], realHigh[i]);
    }
    if (Pstream::master()) std::fclose(file1);
    if (Pstream::master())
    {
        FILE* anchorFile = fopenOrDie("Output/terrain-anchor-diagnostics.tsv", "w");
        std::fprintf(anchorFile, "anchor_version\tturbine\towner_rank\towner_patch_face\towner_triangle\tglobal_intersecting_triangles\tanchor_x_model\tanchor_y_model\tmesh_ground_z_model\thub_z_model\trequested_hub_agl_m\trealized_hub_agl_m\tnearest_bottom_face_centre_distance_model\theight_spread_model\n");
        for (i = 0; i < numTurbine; ++i)
        {
            const windTerrain::TerrainAnchor& anchor = terrainAnchors[i];
            std::fprintf(anchorFile, "%s\t%ld\t%d\t%d\t%d\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n",
                windTerrain::anchorVersion, i + 1, onNode[i], int(anchor.patchFace),
                int(anchor.triangleIndex), int(anchor.globalHitCount), realXYZ[i][0], realXYZ[i][1],
                realXYZ[i][2], realHigh[i], XY[i][2]/scale,
                (realHigh[i] - realXYZ[i][2])/scale, Dxy[i], anchor.heightSpread);
        }
        std::fclose(anchorFile);
    }
#pragma endregion

// =========================== actuator initialization ================== //
    buildActuatorCellSets(mesh);
    sampleReferenceVelocity(mesh, U);
    writeReferenceSamples("Output/Output03-sumU-numU-windU(INIT)");
    if (numTurbine > 0) update_vInlet();
    updateTurbineOperatingPoints();
    writeOperatingPoints("Output/Output04-U-P-Ct-fn(INIT)");
    writeActuatorDiagnostics(mesh, "initial", runTime.value(), true);
    applyActuatorSources(SourceT);

// =========================== add_Roughness() =========================== //
/****************** read_Rou + regularize + calc_Cxy ********************/
#pragma region
    // 1) UTM center from lon/lat (must match rou coordinates)
    LonLat2UTM(lon, lat, utmX, utmY);
    int zoneNo = utmZoneFromLon(lon);
    Info << "UTM zone(inferred)=" << zoneNo << " utmX=" << utmX << " utmY=" << utmY << endl;

    bool roughEnabled = true;

    // 2) pass-1: count groups/points
    in.open("Input/rou");
    if (!in.is_open())
    {
        WarningInFunction << "Cannot open Input/rou. Vegetation roughness disabled." << nl << endl;
        roughEnabled = false;
    }

    long int sGroups = 0;
    long int num_roughness = 0;

    if (roughEnabled)
    {
        std::string line;
        for (int skip=0; skip<4; ++skip) std::getline(in, line);

        while (std::getline(in, line))
        {
            std::istringstream iss(line);
            double val; int cnt=0;
            while (iss >> val) cnt++;
            if (cnt == 0) continue;
            if (cnt == 1) break;         // optional sentinel
            if (cnt == 3)                // group header: z0 h n
            {
                std::istringstream iss2(line);
                double z0,hc; long n;
                iss2 >> z0 >> hc >> n;
                num_roughness += n;
                sGroups++;
            }
        }
        in.close();

        if (num_roughness <= 0)
        {
            WarningInFunction << "rou file contains zero points. Vegetation roughness disabled." << nl << endl;
            roughEnabled = false;
        }
        if (num_roughness > 50000000L)
        {
            FatalErrorInFunction << "Too many roughness points (" << num_roughness << ")." << exit(FatalError);
        }
    }

    // 3) read points and filter to domain window
    double *nx = nullptr, *ny = nullptr, *nh = nullptr;
    long int validrou = 0;

    if (roughEnabled)
    {
        nx = new double[num_roughness];
        ny = new double[num_roughness];
        nh = new double[num_roughness];

        in.open("Input/rou");
        std::string line;
        for (int skip=0; skip<4; ++skip) std::getline(in, line);

        FILE* fdbg = nullptr;
        if (dbgRoughLevel >= 2) fdbg = std::fopen("Output/Output7-rou_x-rou_y-roua-roub", "w");

        const double halfLt = lt / 2.0; // geometry units (meters)
        Info << "Window half-length (m): " << halfLt << " ; sample rou radius ~" << r2 << " m" << endl;

        for (long int g=0; g<sGroups; ++g)
        {
            double z0, hcan; long int np;
            in >> z0 >> hcan >> np;
            for (long int p=0; p<np; ++p)
            {
                double rx, ry;
                in >> rx >> ry;

                double roua = (rx - utmX) * Foam::cos(angle) + (ry - utmY) * Foam::sin(angle);
                double roub = -(rx - utmX) * Foam::sin(angle) + (ry - utmY) * Foam::cos(angle);

                if (dbgRoughLevel >= 2 && fdbg)
                    std::fprintf(fdbg, "%f\t%f\t%f\t%f\n", rx, ry, roua, roub);

                if (std::fabs(roua) <= halfLt && std::fabs(roub) <= halfLt)
                {
                    nx[validrou] = roua;  // meters
                    ny[validrou] = roub;  // meters
                    // generate_rou.py writes canopy heights in physical metres.
                    // Keep n3 in metres; the hf interpolation below applies scale once.
                    nh[validrou] = hcan * vege_times;
                    validrou++;
                }
            }
        }
        if (fdbg) std::fclose(fdbg);
        in.close();

        Info << "rou文件中数据总数(过滤前): " << num_roughness
             << " 落入计算域窗口(validrou): " << validrou << endl;

        if (validrou == 0)
        {
            WarningInFunction << "No rou points fall inside domain window. Vegetation roughness disabled." << nl << endl;
            roughEnabled = false;
            delete[] nx; delete[] ny; delete[] nh;
            nx = ny = nh = nullptr;
        }
    }

    // 4) regular grid accumulation (nearest node)
    double *n1 = nullptr, *n2 = nullptr, *n3 = nullptr;
    long int actual_nodes = 0, null_m = 0, null_n = 0;
    double DT_R = 0.0;

    if (roughEnabled)
    {
        DT_R = lt + cee - std::fmod(lt, cee);
        null_m = (long int)(DT_R / cee);
        null_n = null_m;
        actual_nodes = (null_m + 1) * (null_n + 1);

        n1 = new double[actual_nodes];
        n2 = new double[actual_nodes];
        n3 = new double[actual_nodes];

        for (i = 0; i < actual_nodes; i++)
        {
            n1[i] = (i - i / (null_m + 1) * (null_n + 1)) * cee - DT_R / 2.0; // geometry units
            n2[i] = (i / (null_n + 1)) * cee - DT_R / 2.0;
            n3[i] = 0.0;
        }

        for (long int mr = 0; mr < validrou; mr++)
        {
            double best = lt * lt;
            long int bestIdx = 0;
            for (long int ii = 0; ii < actual_nodes; ii++)
            {
                double dxl = (n1[ii] - nx[mr]);
                double dyl = (n2[ii] - ny[mr]);
                double d2 = dxl*dxl + dyl*dyl;
                if (d2 < best) { best = d2; bestIdx = ii; }
            }
            n3[bestIdx] = nh[mr]; // pre-scale
        }

        if (dbgRoughLevel >= 2)
        {
            file1 = fopenOrDie("Output/Output8-n1-n2-n3", "w");
            for (long int ii = 0; ii < actual_nodes; ii++)
                std::fprintf(file1, "%f\t%f\t%f\n", n1[ii], n2[ii], n3[ii]);
            std::fclose(file1);
        }

        delete[] nx; delete[] ny; delete[] nh;
        nx = ny = nh = nullptr;
    }

    // 5) build Cxy
    if (roughEnabled)
    {
        double h0, hf;
        Cxy    *= 0.0;
        RoughT *= 0.0;

        // --- Start of Modification ---
        const polyPatch& botPatch = mesh.boundaryMesh()[botID];
        const label numFace = botPatch.size();
        const label patchStart = botPatch.start();
        const labelUList& faceOwner = mesh.faceOwner();

        double *faceX = new double[numFace];
        double *faceY = new double[numFace];
        double *faceZ = new double[numFace];

        const double cel = cee * scale;
        const double DT_RS = DT_R * scale;
        double maxFaceZ = -h * scale;

        label cnt = 0;
        forAll(botPatch, faceI)
        {
            const label meshFaceI = patchStart + faceI;
            faceX[cnt] = mesh.faceCentres()[meshFaceI].x();
            faceY[cnt] = mesh.faceCentres()[meshFaceI].y();
            faceZ[cnt] = mesh.faceCentres()[meshFaceI].z();
            if (faceZ[cnt] > maxFaceZ) maxFaceZ = faceZ[cnt];
            cnt++;
        }
        // --- End of Modification ---

        if (Pstream::master())
        {
            Info << "=== 粗糙度计算调试信息 ===" << endl;
            Info << "scale=" << scale << " cee=" << cee
                 << " cel=" << cel << " DT_R=" << DT_R << " DT_RS=" << DT_RS << endl;
            Info << "r12=" << r12 << " r12*scale=" << r12*scale << endl;
            Info << "Cd=" << Cd << " lad_max=" << lad_max << " vege_times=" << vege_times << endl;
            Info << "numFace=" << numFace << endl;
        }

        long int hf_nonzero_count = 0;
        long int height_match_count = 0;
        long int face_cell_match_count = 0;
        double max_hf = 0.0, min_hf = 1e20;
        double max_h0 = 0.0, min_h0 = 1e20;

        // --- Start of Modification ---
        for (label fi = 0; fi < numFace; fi++)
        {
            const double face_radius_sq = faceX[fi]*faceX[fi] + faceY[fi]*faceY[fi];
            const double radius_limit_sq = (r12 * scale) * (r12 * scale);

            if (face_radius_sq < radius_limit_sq)
            {
                long int aa = (long int)std::ceil( (faceX[fi] + DT_RS/2.0) / cel );
                long int bb = (long int)std::ceil( (faceY[fi] + DT_RS/2.0) / cel );
                long int k1 = (bb - 1) * (null_n + 1) + (aa - 1);
                long int k2 = k1 + 1;
                long int k3 = k1 + (null_n + 1);
                long int k4 = k3 + 1;

                if (k1 >= 0 && k4 < (long)actual_nodes)
                {
                    double px = faceX[fi] - n1[k1] * scale;
                    double py = faceY[fi] - n2[k1] * scale;

                    double linear = std::fabs( aa - (faceX[fi] + DT_RS/2.0)/cel )
                                  + std::fabs( bb - (faceY[fi] + DT_RS/2.0)/cel );

                    const double n3k1 = n3[k1];
                    const double n3k2 = n3[k2];
                    const double n3k3 = n3[k3];
                    const double n3k4 = n3[k4];

                    if (linear < 1.0)
                    {
                        hf = scale * ( n3k1
                            - (n3k1 - n3k2)/cel * px
                            - (n3k1 - n3k3)/cel * py );
                    }
                    else
                    {
                        hf = scale * ( n3k3 + n3k2 - n3k4
                            - (n3k3 - n3k4)/cel * px
                            - (n3k2 - n3k4)/cel * py );
                    }
                }
                else
                {
                    hf = 0.0;
                    if (dbgRoughLevel >= 1 && Pstream::master())
                    {
                        Info << "Index OOB: face=" << fi << " (aa,bb)=(" << aa << "," << bb
                             << ") nodes=" << actual_nodes << endl;
                    }
                }
            }
            else
            {
                hf = 0.0;
            }

            if (hf > 0.0)
            {
                hf_nonzero_count++;
                if (hf > max_hf) max_hf = hf;
                if (hf < min_hf) min_hf = hf;
            }

            // Directly get the owner cell for this bottom face
            label cellI = faceOwner[patchStart + fi];
            face_cell_match_count++;

            // faceZ is the actual bot face centre, already in scaled mesh coordinates.
            // The old (2*z-delta)*h/(2*h-delta) expression inverted a first-layer
            // CELL centre to recover terrain elevation; applying it to a face
            // centre lowered the ground twice. Compare this vertical offset to hf.
            h0 = mesh.C()[cellI].z() - faceZ[fi];

            if (h0 > max_h0) max_h0 = h0;
            if (h0 < min_h0) min_h0 = h0;

            if (h0 >= 0.0 && hf > 0.0 && h0 <= hf)
            {
                height_match_count++;

                double roughness_value =
                    (1.0/scale) * Cd * lad_max
                    * Foam::exp( - (h0/hf - 0.5)*(h0/hf - 0.5) * 10.0 );

                if (roughness_value < 0) roughness_value = 0.0;

                Cxy[cellI].x() = roughness_value;
                Cxy[cellI].y() = roughness_value;
                Cxy[cellI].z() = roughness_value;
            }
        }
        // --- End of Modification ---

        // global reductions
        long int hf_nonzero_count_g   = returnReduce(hf_nonzero_count,   sumOp<long int>());
        long int height_match_count_g = returnReduce(height_match_count, sumOp<long int>());
        long int face_cell_match_count_g = returnReduce(face_cell_match_count, sumOp<long int>());
        double max_hf_g = returnReduce(max_hf, maxOp<double>());
        double min_hf_g = returnReduce(min_hf, minOp<double>());
        double max_h0_g = returnReduce(max_h0, maxOp<double>());
        double min_h0_g = returnReduce(min_h0, minOp<double>());

        if (Pstream::master())
        {
            Info << "=== 粗糙度计算统计 ===" << endl;
            Info << "非零植被高度的面数: " << hf_nonzero_count_g << endl;
            Info << "面-网格匹配数: " << face_cell_match_count_g << endl;
            Info << "高度条件匹配数: " << height_match_count_g << endl;
            Info << "植被高度范围: [" << min_hf_g << ", " << max_hf_g << "]" << endl;
            Info << "网格高度范围: [" << min_h0_g << ", " << max_h0_g << "]" << endl;
        }

        delete[] n1; delete[] n2; delete[] n3;
        delete[] faceX; delete[] faceY; delete[] faceZ;
    }
#pragma endregion

#pragma region Debug_Roughness_Statistics
    {
        long int roughCells = 0, totalCells = 0;
        double maxCxy = 0.0, avgCxy = 0.0, totalCxy = 0.0;

        forAll(mesh.cells(), cellI)
        {
            totalCells++;
            if (Cxy[cellI].x() > 0.0)
            {
                roughCells++;
                totalCxy += Cxy[cellI].x();
                if (Cxy[cellI].x() > maxCxy) maxCxy = Cxy[cellI].x();
            }
        }

        roughCells = returnReduce(roughCells, sumOp<long int>());
        totalCells = returnReduce(totalCells, sumOp<long int>());
        totalCxy   = returnReduce(totalCxy,   sumOp<double>());
        maxCxy     = returnReduce(maxCxy,     maxOp<double>());

        if (roughCells > 0) avgCxy = totalCxy / roughCells;

        if (Pstream::master())
        {
            Info << "=== 粗糙度统计信息 ===" << endl;
            Info << "总网格数: " << totalCells << endl;
            Info << "受粗糙度影响的网格数: " << roughCells << endl;
            Info << "粗糙度覆盖率: " << (double)roughCells/totalCells*100.0 << "%" << endl;
            Info << "最大粗糙度系数: " << maxCxy << endl;
            Info << "平均粗糙度系数: " << avgCxy << endl;
        }

        if (dbgRoughLevel >= 2)
        {
            file1 = fopenOrDie("Output/Debug_Roughness_Distribution.txt", "w");
            std::fprintf(file1, "CellID\tX\tY\tZ\tCxy\n");
            forAll(mesh.cells(), cellI)
            {
                if (Cxy[cellI].x() > 0.0)
                {
                    std::fprintf(file1, "%d\t%f\t%f\t%f\t%f\n",
                        cellI, mesh.C()[cellI].x(), mesh.C()[cellI].y(), mesh.C()[cellI].z(), Cxy[cellI].x());
                }
            }
            std::fclose(file1);
        }
    }
#pragma endregion

#pragma region Build_Canopy_Turbulence_Fields
    {
        const scalar safeCd = Foam::max(scalar(1e-6), scalar(Cd));

        roughCd = dimensionedScalar("zero", dimless, 0.0);
        roughLAD = dimensionedScalar("zero", dimensionSet(0, -1, 0, 0, 0, 0, 0), 0.0);

        forAll(mesh.cells(), cellI)
        {
            if (Cxy[cellI].x() > 0.0)
            {
                // Match the custom momentum sink magnitude:
                // Cd*LAD*|U| ~= 0.5*rho*Cxy*|U|
                roughCd[cellI] = safeCd;
                roughLAD[cellI] = 0.5 * p_air * Cxy[cellI].x() / safeCd;
            }
        }

        if (Pstream::master())
        {
            Info << "Constructed roughCd/roughLAD fields for canopy turbulence closure." << endl;
        }
    }
#pragma endregion

    argList::addNote("Steady-state solver for incompressible, turbulent flows.");
    turbulence->validate();

    Info << "\nStarting time loop\n" << endl;

    while (simple.loop())
    {
        Info << "Time = " << runTime.timeName() << nl << endl;

        {
            // Roughness body force
            forAll(mesh.cells(), cellI)
            {
                scalar Um = Foam::sqrt(U[cellI].x()*U[cellI].x() + U[cellI].y()*U[cellI].y() + U[cellI].z()*U[cellI].z());
                U_mag[cellI] = Um;
                const scalar roughDiagCoeff = Foam::max(0.0, 0.5 * 1.22 * Um * Cxy[cellI].x());
                RoughDiag[cellI] = roughDiagCoeff;
                RoughT[cellI].x() = -roughDiagCoeff * U[cellI].x();
                RoughT[cellI].y() = -roughDiagCoeff * U[cellI].y();
                RoughT[cellI].z() = -roughDiagCoeff * U[cellI].z();
            }

            sampleReferenceVelocity(mesh, U);
            updateTurbineOperatingPoints();
            applyActuatorSources(SourceT);
            writeActuatorDiagnostics(mesh, "pre_solve", runTime.value());

#include "UEqn.H"
#include "pEqn.H"
        }

        laminarTransport.correct();
        turbulence->correct();

        runTime.write();
        runTime.printExecutionTime(Info);
    }

    // Preserve the final iterate even when it is not on writeInterval. The
    // exported volume and the final turbine samples must refer to the same U.
    runTime.writeNow();

    // The explicit actuator update above uses the previous iterate. Resample
    // after the last pressure/turbulence correction so delivered turbine data
    // describes the same FINAL U field exported by post-processing.
    sampleReferenceVelocity(mesh, U);
    updateTurbineOperatingPoints();
    writeReferenceSamples("Output/Output05-sumU-numU-windU(ADJUST)");
    writeOperatingPoints("Output/Output06-U-P-Ct-fn(ADJUST)");
    writeActuatorDiagnostics(mesh, "final_resampled", runTime.value());

    char strPath[64], strNum[32];
    std::strcpy(strPath, "Output/txt/");
    numToStr((int)(vInlet + 0.5), strNum, 10);
    std::strcat(strPath, std::strcat(strNum, ".txt"));
    writeOperatingPoints(strPath);

    Info << "End\n" << endl;
    return 0;
}
