import { test, expect } from "@playwright/test";
for (const viewport of [
  { width: 1366, height: 768 },
  { width: 1920, height: 1080 },
]) {
  test(`connected scientific workspaces and persistent exports ${viewport.width}`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/");
    await expect(page.getByText("Local engine connected")).toBeVisible({
      timeout: 30000,
    });
    await page.getByLabel("Search patients").fill("104");
    await page.getByRole("button", { name: "P-104", exact: true }).click();
    await expect(
      page.getByRole("heading", { name: "Patient P-104", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", {
        name: "Digital Tumor Twin Interactive simulation",
      })
      .click();
    await page
      .getByLabel("Patient source", { exact: true })
      .selectOption("synthetic");
    await page.getByLabel("Horizon days", { exact: true }).fill("112");
    await page.getByLabel("Sensitive growth multiplier").fill("1.2");
    await page
      .getByRole("button", { name: "Initialize twin", exact: true })
      .click();
    await expect(
      page.getByLabel("Step simulation", { exact: true }),
    ).toBeEnabled();
    await page.getByLabel("Step simulation", { exact: true }).click();
    await expect(
      page.getByText("DAY 28", { exact: true }).first(),
    ).toBeVisible();
    await page.getByRole("button", { name: "Branch ON vs OFF" }).click();
    await expect(page.getByText("Requested ON", { exact: true })).toBeVisible();
    await page.getByLabel("Pause simulation", { exact: true }).click();
    await expect(
      page.getByLabel("Step simulation", { exact: true }),
    ).toBeDisabled();
    await page
      .getByLabel("Start or resume simulation", { exact: true })
      .click();
    await expect(page.getByText("Running", { exact: true })).toBeVisible();
    await page.getByLabel("Pause simulation", { exact: true }).click();
    await page.getByLabel("Reset simulation", { exact: true }).click();
    await expect(
      page.getByText("DAY 0", { exact: true }).first(),
    ).toBeVisible();
    await page.getByLabel("Step simulation", { exact: true }).click();
    await page
      .getByRole("button", { name: "RL Decision Laboratory Policy evidence" })
      .click();
    await expect(
      page.getByText("Checkpoint action distribution", { exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Next decision", exact: true })
      .click();
    await expect(page.getByText("DECISION AT DAY 56", {exact:true})).toBeVisible();
    await page.getByLabel("Decision timeline").fill("1");
    await expect(
      page.getByText("DECISION AT DAY 28", { exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", {
        name: "Digital Tumor Twin Interactive simulation",
      })
      .click();
    await page.getByLabel("Simulation timeline").fill("0");
    await expect(
      page.getByText("DAY 0", { exact: true }).first(),
    ).toBeVisible();
    await page
      .getByLabel("Decision policy", { exact: true })
      .selectOption("continuous");
    await expect(page.getByText(/Draft parameters changed/)).toBeVisible();
    await expect(
      page.getByLabel("Step simulation", { exact: true }),
    ).toBeDisabled();
    await page
      .getByRole("button", { name: "Initialize twin", exact: true })
      .click();
    await page.getByLabel("Step simulation", { exact: true }).click();
    await expect(page.getByRole("alert")).toHaveCount(0);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBeTruthy();
    await page.screenshot({
      path: `test-results/twin-${viewport.width}.png`,
      fullPage: false,
    });
    await page.getByLabel("Replay simulation", { exact: true }).click();
    await expect(
      page.getByText("DAY 112", { exact: true }).first(),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Strategy Arena Matched comparisons" })
      .click();
    await page.getByLabel(/Include growth.*sensitivity/).uncheck();
    await page
      .getByRole("button", { name: "Run comparison", exact: true })
      .click();
    await expect(
      page.getByText("completed", { exact: true }).first(),
    ).toBeVisible({ timeout: 30000 });
    await expect(
      page.getByRole("link", { name: "CSV", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Experiment Center Reproducible research" })
      .click();
    await page
      .getByLabel("Experiment name", { exact: true })
      .fill("Browser verified study");
    await page.getByLabel("Experiment seeds", { exact: true }).fill("17");
    const saving = page.waitForResponse(r=>r.url().endsWith('/api/experiments')&&r.request().method()==='POST');
    await page.getByRole("button", { name: "Save & run experiment", exact: true }).click();
    const savedId = (await (await saving).json()).id;
    await expect(page.getByRole('link',{name:'CSV',exact:true})).toHaveAttribute('href', `/api/experiments/${savedId}/export?format=csv`);
    await expect(page.getByRole('button',{name:'Reproduce saved run'})).toBeVisible({timeout:30000});
    const download = page.waitForEvent("download");
    await page.getByRole("link", { name: "CSV", exact: true }).click();
    expect((await download).suggestedFilename()).toMatch(/oncoevo.*\.csv/);
    const reproducing=page.waitForResponse(r=>r.url().endsWith('/reproduce')&&r.request().method()==='POST');
    await page.getByRole("button", { name: "Reproduce saved run" }).click();
    const replayId=(await(await reproducing).json()).id;
    await expect(page.getByRole('link',{name:'CSV',exact:true})).toHaveAttribute('href', `/api/experiments/${replayId}/export?format=csv`);
    await expect(page.getByRole('button',{name:'Reproduce saved run'})).toBeVisible({timeout:30000});
    await expect(page.getByRole("alert")).toHaveCount(0);
    expect(errors).toEqual([]);
    await page.screenshot({
      path: "test-results/experiments-studio.png",
      fullPage: true,
    });
  });
}
test("mobile layout stays within viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByText("Validated recorded patient series")).toBeVisible(
    { timeout: 30000 },
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
});
