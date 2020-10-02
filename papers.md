<table>
    <thead>
        <tr>
            <th>Paper</th>
            <th>Main ideas and results</th>
            <th>Other facts</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                Zadrozny В., Elkan C. <a href="https://cseweb.ucsd.edu/~elkan/calibrated.pdf">«Obtaining  calibrated  probability  estimates  from  decision  trees  and naive bayesian classifiers»</a>, 2001
            </td>
            <td><ul>
                <li>Applying <b>m-estimation</b> (generalized Laplace smoothing) to correct most distant probabilities and shift towards the base rate(standard one, based on <a href="https://en.wikipedia.org/wiki/Rule_of_succession">rule of succession</a> adjusts estimates closer to 1/2 which is not reasonable in unbalanced classes). <a href="https://www.researchgate.net/publication/220838515_Estimating_Probabilities_A_Crucial_Task_in_Machine_Learning">details</a></li>
                <li>[DT] <b>Curtailment</b> - taking into account not only class frequencies of the leaves but also of the closest ancestors. can be implemented with unconventional pruning</li>
                <li><b>Binning</b> (histogram method): training examples are sorted by scores into subsets of equal size, and the <i>estimated corrected probability</i> for the test object is now the "accuracy" inside the bin it belongs to</li>
                <li><b>Evaluation metrics</b> of calibration quality: MSE and log-loss are more suitable than lift charts and "profit achieved" (specific problem-related metric)</li>
            </td></ul>
            <td>
                <ul>
                    <li>DTs' <em>predict proba</em> is just the raw training frequency of final leaf. That's not reliable: 
                        <ol>
                            <li>such frequencies are usually shifted towards 0 or 1 since DTs strive to have homogeneous leaves;</li>
                            <li>without pruning, leaf «capacity» can be small</li>
                        </ol></li>
                    <li>Standard DT pruning (maximizing accuracy) methods do not improve quality of probability estimation</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>
