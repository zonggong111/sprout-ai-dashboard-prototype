/*
 * Independent fictional dataset for visual-design review only.
 * No workbook values, source identifiers, original questions, or internal records are used.
 */
(function () {
  const topicPlan = [
    ["Healthy Eating / Nutrition", 54],
    ["Physical Activity", 48],
    ["Sleep", 42],
    ["Screen Use", 34],
    ["Growth & Development", 24],
    ["Health Plan / Navigation", 18],
    ["Wellbeing", 12],
    ["Other / Unclassified", 8]
  ];
  const faqPlan = [
    ["Building a balanced lunchbox", 14, "Healthy Eating / Nutrition"],
    ["Calmer bedtime routines", 13, "Sleep"],
    ["Reaching the daily activity goal", 12, "Physical Activity"],
    ["Managing requests for more screen time", 11, "Screen Use"],
    ["Choosing lower-sugar drinks", 10, "Healthy Eating / Nutrition"],
    ["Finding the child health plan", 9, "Health Plan / Navigation"],
    ["Supporting worries about school", 8, "Wellbeing"],
    ["Understanding healthy growth ranges", 7, "Growth & Development"]
  ];
  const conditionalPlan = [
    ["Medical guidance", 10],
    ["Bullying & online safety", 7],
    ["Emotional wellbeing", 5],
    ["Weight-management advice", 5],
    ["Sensitive health topic", 3]
  ];
  const unablePlan = [
    ["Health Plan / Navigation", 7],
    ["Healthy Eating / Nutrition", 5],
    ["Sleep", 4],
    ["Screen Use", 4],
    ["Growth & Development", 3],
    ["Physical Activity", 2],
    ["Wellbeing", 2],
    ["Other / Unclassified", 1]
  ];
  const issueByTopic = {
    "Health Plan / Navigation": ["Navigation / Guidance Issue", "Improve in-product wayfinding"],
    "Healthy Eating / Nutrition": ["Retrieval / Knowledge Issue", "Review nutrition content coverage"],
    "Sleep": ["Incomplete Answer", "Add a clearer sleep guidance response"],
    "Screen Use": ["Unsupported Query", "Define supported screen-use guidance"],
    "Growth & Development": ["Retrieval / Knowledge Issue", "Review growth-reference content"],
    "Physical Activity": ["Incomplete Answer", "Add activity examples and escalation guidance"],
    "Wellbeing": ["Safety Handling", "Review safe wellbeing response pattern"],
    "Other / Unclassified": ["Other", "Route for taxonomy review"]
  };

  const expandedTopics = topicPlan.flatMap(([topic, count]) => Array.from({ length: count }, (_, index) => ({ topic, seed: `${topic}-${index}` })));
  const turns = expandedTopics
    .map((item, index) => ({ ...item, order: (index * 97) % expandedTopics.length }))
    .sort((a, b) => a.order - b.order)
    .map((item, index) => ({
      turnId: `DEMO-TURN-${String(index + 1).padStart(3, "0")}`,
      topic: item.topic,
      faqGroup: null,
      guardrailBehaviour: "Allow",
      guardrailTopic: null,
      unableToAnswer: false,
      reviewSignal: "Routine response",
      reviewOutcome: "Not flagged",
      reviewStatus: "No review required",
      issueType: "Not applicable",
      proposedAction: "No action proposed"
    }));

  faqPlan.forEach(([label, count, topic]) => {
    turns.filter(turn => turn.topic === topic && !turn.faqGroup).slice(0, count).forEach(turn => { turn.faqGroup = label; });
  });

  let conditionalCursor = 0;
  conditionalPlan.forEach(([label, count]) => {
    let assigned = 0;
    while (assigned < count) {
      const turn = turns[(conditionalCursor * 7 + 3) % turns.length];
      conditionalCursor += 1;
      if (turn.guardrailBehaviour === "Conditional") continue;
      turn.guardrailBehaviour = "Conditional";
      turn.guardrailTopic = label;
      assigned += 1;
    }
  });

  unablePlan.forEach(([topic, count]) => {
    turns.filter(turn => turn.topic === topic).slice(0, count).forEach(turn => { turn.unableToAnswer = true; });
  });

  turns.forEach((turn, index) => {
    const flagged = turn.unableToAnswer || turn.guardrailBehaviour === "Conditional" || Boolean(turn.faqGroup);
    if (!flagged) return;
    turn.reviewSignal = turn.unableToAnswer ? "Unable to answer" : turn.guardrailBehaviour === "Conditional" ? "Conditional guardrail" : "Repeated question";
    const outcomeIndex = (index * 7 + 3) % 10;
    if (outcomeIndex < 2) {
      turn.reviewOutcome = "Flagged for Review";
      turn.reviewStatus = "Flagged for Review";
    } else if (outcomeIndex < 5) {
      turn.reviewOutcome = "Pending Human Review";
      turn.reviewStatus = "Pending Human Review";
    } else if (outcomeIndex < 8) {
      turn.reviewOutcome = "Reviewed — No Confirmed Issue";
      turn.reviewStatus = "Review completed";
    } else {
      turn.reviewOutcome = "Confirmed Issue";
      turn.reviewStatus = "Review completed";
      [turn.issueType, turn.proposedAction] = issueByTopic[turn.topic];
    }
  });

  const markedTurns = turns.slice(0, 120);
  const records = [];
  const cleanDirection = value => value === "Pass" ? "Pass" : "Fail";
  markedTurns.forEach((turn, index) => {
    const caseNumber = index + 1;
    const agent = caseNumber <= 48 ? "Recommendation" : caseNumber <= 88 ? "Information" : "Guardrail";
    const dimensions = agent === "Guardrail" ? ["Safety"] : ["Accuracy", "Quality"];
    dimensions.forEach((dimension, dimensionIndex) => {
      const judge = (caseNumber * 3 + dimensionIndex * 4) % 11 < 8 ? "Pass" : "Fail";
      const humanSeed = (caseNumber * 7 + dimensionIndex * 3) % 17;
      const human = humanSeed < 13 ? "Pass" : humanSeed === 16 ? "Fail - Known Issue" : "Fail";
      const different = cleanDirection(judge) !== cleanDirection(human);
      const leadRequired = different && (caseNumber + dimensionIndex) % 3 !== 0;
      const lead = leadRequired ? ((caseNumber + dimensionIndex) % 4 === 0 ? cleanDirection(judge) : cleanDirection(human)) : null;
      const consolidated = lead || human;
      const discrepancy = judge === "Pass" && cleanDirection(human) === "Fail"
        ? "Judge Pass / Human Fail"
        : judge === "Fail" && cleanDirection(human) === "Pass"
          ? "Judge Fail / Human Pass"
          : "Agree";
      records.push({
        caseId: `DEMO-${String(caseNumber).padStart(3, "0")}`,
        turnId: turn.turnId,
        agent,
        dimension,
        judge,
        human,
        lead,
        consolidated,
        discrepancy,
        leadOverride: Boolean(lead && cleanDirection(lead) !== cleanDirection(human)),
        reviewStatus: lead ? "Completed adjudication" : different ? "Pending adjudication" : "No adjudication required",
        topic: turn.topic,
        faqGroup: turn.faqGroup,
        guardrailTopics: turn.guardrailTopic ? [turn.guardrailTopic] : [],
        unableToAnswer: turn.unableToAnswer,
        reviewSignal: turn.reviewSignal,
        reviewOutcome: turn.reviewOutcome,
        issueType: turn.issueType,
        proposedAction: turn.proposedAction
      });
    });
  });

  const groupOrder = [
    ["Recommendation", "Accuracy"], ["Recommendation", "Quality"],
    ["Information", "Accuracy"], ["Information", "Quality"], ["Guardrail", "Safety"]
  ];
  const outcomeCounts = rows => ({
    pass: rows.filter(row => row === "Pass").length,
    fail: rows.filter(row => row === "Fail").length,
    known: rows.filter(row => row === "Fail - Known Issue").length
  });
  const withRates = counts => {
    const total = counts.pass + counts.fail + counts.known;
    const adjustedTotal = counts.pass + counts.fail;
    return { ...counts, rate: total ? counts.pass / total * 100 : 0, adjusted: adjustedTotal ? counts.pass / adjustedTotal * 100 : 0 };
  };
  const calibration = groupOrder.map(([agent, dimension]) => {
    const rows = records.filter(row => row.agent === agent && row.dimension === dimension);
    return {
      agent, dimension, sample: rows.length,
      before: withRates(outcomeCounts(rows.map(row => row.human))),
      after: withRates(outcomeCounts(rows.map(row => row.consolidated))),
      overrides: rows.filter(row => row.leadOverride).length
    };
  });
  const discrepancies = groupOrder.map(([agent, dimension]) => {
    const rows = records.filter(row => row.agent === agent && row.dimension === dimension);
    const before = rows.map(row => [cleanDirection(row.judge), cleanDirection(row.human)]);
    const after = rows.map(row => [cleanDirection(row.judge), cleanDirection(row.consolidated)]);
    const count = (pairs, judge, human) => pairs.filter(pair => pair[0] === judge && pair[1] === human).length;
    return {
      agent, dimension, sample: rows.length,
      agree: before.filter(pair => pair[0] === pair[1]).length,
      beforeJudgeFailHumanPass: count(before, "Fail", "Pass"),
      beforeJudgePassHumanFail: count(before, "Pass", "Fail"),
      afterJudgeFailHumanPass: count(after, "Fail", "Pass"),
      afterJudgePassHumanFail: count(after, "Pass", "Fail")
    };
  });
  const aggregate = (values, key) => Object.entries(values.reduce((result, item) => {
    const label = item[key];
    if (label) result[label] = (result[label] || 0) + 1;
    return result;
  }, {})).map(([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value);

  window.ReportData = {
    source: {
      workbook: "Independent synthetic scenario model",
      evaluationPeriod: "Illustrative Beta Round 2 review period",
      evaluationGenerated: "Synthetic visual proposal",
      markingPeriod: "Illustrative marking cycle",
      sourcesUsed: ["Synthetic conversation scenarios", "Synthetic evaluation outcomes", "Synthetic marking assignments"]
    },
    turns,
    evaluationScope: { totalRecords: 240, processed: 236, skippedMissingProfile: 4, judgeErrors: 0, humanMarkedTurns: 120, assignedParents: 40, completedParents: 34 },
    conversation: {
      guardrailBehaviour: { Allow: turns.filter(turn => turn.guardrailBehaviour === "Allow").length, Conditional: turns.filter(turn => turn.guardrailBehaviour === "Conditional").length },
      conditionalTopics: aggregate(turns.filter(turn => turn.guardrailBehaviour === "Conditional"), "guardrailTopic"),
      topicDistribution: aggregate(turns, "topic"),
      faqGroups: aggregate(turns.filter(turn => turn.faqGroup), "faqGroup").slice(0, 8),
      unableByTopic: aggregate(turns.filter(turn => turn.unableToAnswer), "topic"),
      reviewOutcomes: aggregate(turns.filter(turn => turn.reviewOutcome !== "Not flagged"), "reviewOutcome")
    },
    calibration,
    discrepancies,
    markerProgress: [
      { marker: "Marker A", assignedParents: 12, completedParents: 11, completedTurns: 34 },
      { marker: "Marker B", assignedParents: 10, completedParents: 9, completedTurns: 31 },
      { marker: "Marker C", assignedParents: 10, completedParents: 8, completedTurns: 29 },
      { marker: "Marker D", assignedParents: 8, completedParents: 6, completedTurns: 26 }
    ]
  };
  window.ReportCaseData = {
    meta: {
      markedTurns: markedTurns.length,
      caseDimensionRecords: records.length,
      comparableAssessments: records.length,
      pendingAssessments: records.filter(record => record.reviewStatus === "Pending adjudication").length,
      completedAdjudications: records.filter(record => record.reviewStatus === "Completed adjudication").length,
      noAdjudicationRequired: records.filter(record => record.reviewStatus === "No adjudication required").length,
      caseIdMethod: "Fictional sequential identifiers created solely for the visual proposal."
    },
    records
  };
}());
