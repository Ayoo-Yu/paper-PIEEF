    // Momentum predictor

    MRF.correctBoundaryVelocity(U);

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(U)
      + turbulence->divDevReff(U)
      + fvm::Sp(RoughDiag, U)
     ==
        fvOptions(U)+SourceT
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (simple.momentumPredictor())
{
    // --- DEBUGGING CODE START (METHOD 3) ---
    // 在求解之前，计算并打印将要添加的源项的最大值
    // 我们检查 RoughT 和 SourceT (涡轮机)
    scalar maxRoughT = gMax(mag(RoughT)());
scalar maxSourceT = gMax(mag(SourceT)());

    // 只在每次迭代的开始打印一次
    if (Pstream::master())
    {
        Info << "UEqn Sources Check -> Max |RoughT|: " << maxRoughT
             << " , Max |SourceT|: " << maxSourceT << endl;
    }
    // --- DEBUGGING CODE END (METHOD 3) ---

    solve(UEqn == -fvc::grad(p));

    fvOptions.correct(U);
}
