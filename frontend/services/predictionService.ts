

import type {
  PredictionRequest,
  PredictionResponse,
} from "../types/prediction";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8080";

export async function predictNextSession(
  request: PredictionRequest,
): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/api/v1/predictions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Prediction API error (${response.status}): ${detail}`,
    );
  }

  return response.json() as Promise<PredictionResponse>;
}
