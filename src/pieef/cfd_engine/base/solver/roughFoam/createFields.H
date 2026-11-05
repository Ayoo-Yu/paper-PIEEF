Info << "Reading field p\n" << endl;
volScalarField p(
    IOobject(
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading field U\n" << endl;
volVectorField U(
    IOobject(
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading custom fields for roughness and turbine modeling\n" << endl;

volVectorField Cxy(
    IOobject(
        "Cxy",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,        // 正确：你有初始文件
        IOobject::AUTO_WRITE),
    mesh);

volScalarField roughCd(
    IOobject(
        "roughCd",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("zero", dimless, 0.0));

volScalarField roughLAD(
    IOobject(
        "roughLAD",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("zero", dimensionSet(0, -1, 0, 0, 0, 0, 0), 0.0));

volScalarField U_mag(
    IOobject(
        "U_mag",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,        // 正确：你有初始文件
        IOobject::AUTO_WRITE),
    mesh);

volVectorField RoughT(
    IOobject(
        "RoughT",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,        // 正确：你有初始文件
        IOobject::AUTO_WRITE),
    mesh);

volScalarField RoughDiag(
    IOobject(
        "RoughDiag",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("zero", dimensionSet(0, 0, -1, 0, 0, 0, 0), 0.0));

volVectorField SourceT(
    IOobject(
        "SourceT",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,        // 新增：因为你也有SourceT初始文件
        IOobject::AUTO_WRITE),
    mesh);

#include "createPhi.H"

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, simple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());

singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence(
    incompressible::turbulenceModel::New(U, phi, laminarTransport));

#include "createMRF.H"
#include "createFvOptions.H"
