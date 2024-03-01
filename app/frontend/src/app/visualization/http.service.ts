import { Injectable } from '@angular/core'
import { HttpClient, HttpEvent, HttpHeaders, HttpRequest } from '@angular/common/http'
import { Observable, Subject } from 'rxjs'
import { finalize } from 'rxjs/operators'

@Injectable({
    providedIn: 'root',
})
export class HttpService {
    httpOptions = {
        headers: new HttpHeaders({ 'Content-Type': 'application/json' }),
    }

    private activeRequests: number = 0
    private activeRequestsSubject: Subject<boolean> = new Subject<boolean>()

    constructor(private http: HttpClient) {}

    get<T>(url: string, options?: any): Observable<HttpEvent<T>> {
        this.notifyPendingRequest()
        return this.http.get<T>(url, options).pipe(finalize(() => this.notifyCompletedRequest()))
    }

    post<T>(url: string, body: any, options?: any): Observable<HttpEvent<T>> {
        this.notifyPendingRequest()
        return this.http.post<T>(url, body, options).pipe(finalize(() => this.notifyCompletedRequest()))
    }

    // Add more methods for other HTTP verbs if needed

    private notifyPendingRequest(): void {
        this.activeRequests++
        this.activeRequestsSubject.next(true)
    }

    private notifyCompletedRequest(): void {
        this.activeRequests--
        if (this.activeRequests === 0) {
            this.activeRequestsSubject.next(false)
        }
    }

    isRequestPending(): Observable<boolean> {
        return this.activeRequestsSubject.asObservable()
    }
}
