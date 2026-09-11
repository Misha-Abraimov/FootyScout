// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Navbar } from "./Navbar";

vi.mock("next/navigation", () => ({ usePathname: () => "/" }));

afterEach(cleanup);

describe("Navbar", () => {
  it("uses the FootyScout monogram and brand name", () => {
    render(<Navbar />);

    expect(screen.getByText("FS")).toBeTruthy();
    expect(screen.getByText("FootyScout")).toBeTruthy();
    expect(screen.queryByText("SL")).toBeNull();
  });
});
