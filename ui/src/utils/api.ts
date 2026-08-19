import axios from 'axios';
import { createGlobalState } from 'react-global-hooks';

export const isAuthorizedState = createGlobalState(false);

export const apiClient = axios.create();

// Auth note: no Authorization header is attached here on purpose. The browser
// sends the AITK_SESSION cookie automatically (same-origin), and the
// password itself is never stored client-side - login is an Ed25519
// challenge-response (see src/utils/authKey.ts).

// Add a response interceptor to handle 401 errors
apiClient.interceptors.response.use(
  response => response, // Return successful responses as-is
  error => {
    // Check if the error is a 401 Unauthorized
    if (error.response && error.response.status === 401) {
      // Session expired/invalid; the UI re-gates via /api/auth.
      isAuthorizedState.set(false);
    }

    // Reject the promise so calling code can still catch it
    return Promise.reject(error);
  },
);
